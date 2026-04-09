/*! A convenient, safe, and performant API for atomic file I/O */

use rustix::{
    fs::{Mode, OFlags, copy_file_range, ftruncate, ioctl_ficlone, open},
    io::Errno,
    mm::{MapFlags, MremapFlags, MsyncFlags, ProtFlags, mmap, mremap, msync, munmap},
};
use std::{ffi::c_void, fs::File, io, os::fd::AsFd, path::Path};

/// A snapshot of a file
///
/// ## Reading
///
/// Read the file contents using the `AsRef` impl.  The data you see will
/// reflect the state of the file at the time `open()` was called; writes by other
/// process are not reflected.  In other words, `Atommap` will show you a consistent
/// point-in-time snapshot of the file.
///
/// Data is not loaded eagerly into memory.  It will be read in from disk on demand.
/// For this we rely on the COW capabilities of the underlying filesystem.
///
/// ## Writing
///
/// Write the file contents using the `AsMut` impl.  Writes will be immediately
/// visible to you (when you read this `Atommap`), but will not be visible to
/// other processes reading the file until you call `commit()`.  Once you
/// call commit, all your modifications will be atomically visible to other
/// readers.
///
/// Modifications are being written back to disk all the time (asynchronously
/// by the kernel), so there may be very little I/O left to do when you actually
/// call `commit()`.  "Committing" simply makes the written changes visible
/// (after waiting for writeback to complete).
pub struct Atommap {
    original: File,
    private: File,    // Unlinked; initially a clone of `original`
    ptr: *mut c_void, // Can be null
    len: usize,       // zero iff `ptr` is null
}

unsafe impl Send for Atommap {}
unsafe impl Sync for Atommap {}

fn ficlone_with_fallback(fd_in: impl AsFd, fd_out: impl AsFd, len: usize) -> io::Result<()> {
    // Just use copy_file_range unconditionally?  TODO: Check the performance
    match ioctl_ficlone(&fd_out, &fd_in) {
        Ok(()) => (),
        Err(Errno::OPNOTSUPP) => {
            let mut rem = len;
            while rem > 0 {
                let n = copy_file_range(&fd_in, None, &fd_out, None, rem)?;
                if n == 0 {
                    Err(io::ErrorKind::UnexpectedEof)?;
                }
                if n > rem {
                    panic!()
                }
                rem -= n;
            }
        }
        e => e?,
    }
    Ok(())
}

impl Atommap {
    /// Take a snapshot of the file and map it into memory
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let original = File::options().read(true).write(true).open(path)?;
        let len = original.metadata()?.len() as usize;
        let dir = path.parent().unwrap_or(Path::new("."));
        let private: File =
            open(dir, OFlags::TMPFILE | OFlags::RDWR, Mode::RUSR | Mode::WUSR)?.into();
        ficlone_with_fallback(&original, &private, len)?;

        unsafe {
            let ptr = mmap(
                std::ptr::null_mut(),
                len,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &private,
                0,
            )?;
            Ok(Atommap {
                original,
                private,
                ptr,
                len,
            })
        }
    }

    /// Replace the original file with the contents of the snapshot
    pub fn commit(&mut self) -> io::Result<()> {
        unsafe {
            msync(self.ptr, self.len, MsyncFlags::SYNC)?;
        }
        ficlone_with_fallback(&self.private, &self.original, self.len)?;
        Ok(())
    }

    /// Change the size of the file.  If extending, the extension is filled with zeroes.
    pub fn resize(&mut self, new_len: usize) -> io::Result<()> {
        ftruncate(&self.private, new_len as u64)?;
        unsafe {
            self.ptr = mremap(self.ptr, self.len, new_len, MremapFlags::MAYMOVE)?;
        }
        self.len = new_len;
        Ok(())
    }
}

impl AsRef<[u8]> for Atommap {
    fn as_ref(&self) -> &[u8] {
        if self.len == 0 {
            &[] // core::slice::from_raw_parts rejects (null, 0)
        } else {
            unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }
}

impl AsMut<[u8]> for Atommap {
    fn as_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut [] // core::slice::from_raw_parts rejects (null, 0)
        } else {
            unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
        }
    }
}

impl Drop for Atommap {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                match munmap(self.ptr, self.len) {
                    Ok(()) => (),
                    Err(e) => eprintln!("munmap failed: {e}"),
                }
            }
        }
    }
}
