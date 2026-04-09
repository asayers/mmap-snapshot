use rustix::{
    fs::{Mode, OFlags, copy_file_range, ftruncate, ioctl_ficlone, open},
    io::Errno,
    mm::{MapFlags, MremapFlags, MsyncFlags, ProtFlags, mmap, mremap, msync, munmap},
};
use std::{
    ffi::c_void,
    fs::File,
    io,
    os::fd::AsFd,
    path::{Path, PathBuf},
};

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
    pub fn open(path: PathBuf) -> io::Result<Self> {
        let original = File::options().read(true).write(true).open(&path)?;
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

    pub fn commit(&mut self) -> io::Result<()> {
        unsafe {
            msync(self.ptr, self.len, MsyncFlags::SYNC)?;
        }
        ficlone_with_fallback(&self.private, &self.original, self.len)?;
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
