/*! Safe `mmap()` with **snapshot isolation** and **atomic commits**


## Example

```rust
# use mmap_snapshot::Mmap;
# fn foo() -> std::io::Result<()> {
# let path = std::path::Path::new("/tmp/foo");
# std::fs::write(&path, b"Hello world!")?;
let mut mmap = Mmap::open(&path)?;
assert_eq!(mmap.as_ref(), b"Hello world!");
mmap.as_mut()[6..11].copy_from_slice(b"sekai");
mmap.commit()?;
assert_eq!(std::fs::read_to_string(&path)?, "Hello sekai!");
# Ok(())
# }
```

## Performance

`Atommap::open()` takes a bit longer than `File::open()` (0.1 ms longer on my
machine, although this may depend on filesystem; I'm using XFS).  This is a
one-time cost, and once it's paid there appears to be no performance impact -
ie. it's the same as `pread()`/`pwrite()`/`mmap()` without snapshot isolation.
*/

use rustix::{
    fs::{AtFlags, Mode, OFlags, copy_file_range, ftruncate, ioctl_ficlone, linkat, open},
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

/// Returns whether it fell back
fn ficlone(fd_out: impl AsFd, fd_in: impl AsFd, len: usize) -> io::Result<bool> {
    match ioctl_ficlone(&fd_out, &fd_in) {
        Ok(()) => Ok(false),
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
            Ok(true)
        }
        Err(e) => Err(e.into()),
    }
}

/// A point-in-time snapshot of a file
///
/// The snapshot can be modified and then atomically committed to disk,
/// overwriting the contents of file.
///
/// ## Reading
///
/// Read the file contents using the `AsRef` impl.  The data you see will
/// reflect the state of the file at the time `open()` was called; writes by other
/// process are not reflected.  In other words, `Mmap` will show you a consistent
/// point-in-time snapshot of the file.
///
/// Data is not loaded eagerly into memory.  It will be read in from disk on demand.
/// For this we rely on the COW capabilities of the underlying filesystem.
///
/// ## Writing
///
/// Modify the contents using the `AsMut` impl.  Writes will not be visible
/// to other processes reading the file until you call `commit()`.  Once you
/// call `commit()`, all your modifications will be atomically visible to other
/// readers.  If you drop the `Mmap` without calling `commit()`, your writes
/// will be lost!
///
/// Modifications are written to disk continuously in the background; `commit()`
/// simply waits for writeback to finish, and then makes the written changes
/// visible.
pub struct Mmap {
    original: File,
    private: File, // Unlinked; initially a clone of `original`
    ptr: *mut c_void,
    len: usize,
    path: Option<PathBuf>,
}

unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let original = File::options().read(true).write(true).open(path)?;
        let len = original.metadata()?.len() as usize;
        let dir = path.parent().unwrap_or(Path::new("."));
        let private: File =
            open(dir, OFlags::TMPFILE | OFlags::RDWR, Mode::RUSR | Mode::WUSR)?.into();
        let fellback = ficlone(&private, &original, len)?;

        // SAFETY:
        // > If `ptr` is not null, it must be aligned...
        // `ptr` is null.
        //
        // > If there exist any Rust references referring to the memory region, or if
        // > you subsequently create a Rust reference referring to the resulting region,
        // > it is your responsibility to ensure that the Rust reference invariants are
        // > preserved, including ensuring that the memory is not mutated in a way that
        // > a Rust reference would not expect.
        //
        // I believe this is satified if the the only way to mutate the memory
        // is via this Mmap's `AsMut` impl.  The other way this memory could be
        // mutated is by modifications to the file. Since the file was created
        // with O_TMPFILE, it's impossible to create a new fd for the file via
        // the filesystem. And since we never expose our fd, it's impossible to
        // create a new fd via clone().  Therefore we hold the only fd. So long
        // as _we_ don't modify the file via that fd (which we don't), the file
        // can only be modified via the mmap. This satisfies the requirements.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &private,
                0,
            )?
        };
        assert!(!ptr.is_null());
        Ok(Self {
            original,
            private,
            ptr,
            len,
            path: fellback.then(|| path.to_owned()),
        })
    }

    pub fn commit(&mut self) -> io::Result<()> {
        self.sync()?;
        match &self.path {
            Some(path) => {
                // We can't just copy self.private to self.original, since
                // this would not be atomic. And we need to keep self.private
                // unlinked. So we create a new private file, copy over the
                // contents, and link it.
                let dir = path.parent().unwrap_or(Path::new("."));
                let private2: File =
                    open(dir, OFlags::TMPFILE | OFlags::RDWR, Mode::RUSR | Mode::WUSR)?.into();
                // This is non-atomic but that's fine, since we're holding &mut
                // self and therefore `self.private` can't recieve modifications
                // while the copy is in-progress
                ficlone(&private2, &self.private, self.len)?;
                linkat(&private2, "", rustix::fs::CWD, path, AtFlags::EMPTY_PATH)?;
            }
            None => ioctl_ficlone(&self.original, &self.private)?,
        }
        Ok(())
    }

    pub fn commit_and_close(mut self) -> io::Result<()> {
        match self.path.take() {
            Some(path) => self.link(path),
            None => self.commit(),
        }
    }

    pub fn link(self, path: impl AsRef<Path>) -> io::Result<()> {
        linkat(
            &self.private,
            "",
            rustix::fs::CWD,
            path.as_ref(),
            AtFlags::EMPTY_PATH,
        )?;
        Ok(())
    }

    fn sync(&self) -> io::Result<()> {
        if self.len != 0 {
            unsafe {
                msync(self.ptr, self.len, MsyncFlags::SYNC)?;
            }
        }
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

impl AsRef<[u8]> for Mmap {
    fn as_ref(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl AsMut<[u8]> for Mmap {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }
}

impl Drop for Mmap {
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
