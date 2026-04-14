#![warn(missing_docs)]

/*! **Safe** `mmap()` with **snapshot isolation** and **atomic commits**.

([Linux-only](#os-support), [works best](#performance) on XFS/btrfs.)

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

## Safety

The unsafe thing about mmapping a file is that what you get is volatile memory -
when someone modifies the file, the memory changes.  This is not the way a
respectable `&[u8]` should behave.

Instead of mapping the file directly, our trick is to map a private "snapshot"
of the file which doesn't change, even when the file is modified.
The *only* way to modify the snapshot is via the mmap,
which makes it a valid `&mut [u8]` according to Rust's rules.

<div class="warning">

There are a few crates out there which expose "safe" `mmap()` without doing
anything to ensure that the file isn't externally modified.  These are simply
unsound and should not be used!  If you want to risk UB, that's fine - use
[`memmap2`](https://crates.io/crates/memmap2) and write the `unsafe` yourself.

</div>

## OS support

We make the snapshot by cloning the original file into an unlinked file.
It's impossible for anyone else to modify this file, which is what makes it safe to mmap.
On Linux we use `O_TMPFILE` for this.
I don't know of a race-free way to create an unlinked file on MacOS/Windows;
if one exists, please open an issue to let me know!

## Performance

This crate has the same semantics on all filesystems, but wildly different
performance characteristics.  This table shows whether methods are constant-time
(✅) or linear-time (⏳️) in the size of the file:

Method | XFS | btrfs | ext4
-------|-----|-------|---------
[`open()`][`Mmap::open`]                         | ✅ | ✅ | ⏳️
[`commit()`][`Mmap::commit`]                     | ✅ | ✅ | ⏳️
[`commit_and_close()`][`Mmap::commit_and_close`] | ✅ | ✅ | ✅

See the method docs for more details.

Although many distros now default to reflink-capable filesystems for new
installs[^debian], it will obviously be common to encounter ext4 in the wild for
many years to come.  Be aware that a subset of your users may experience stalls
when mmapping large files.

[^debian]: The major exceptions are Debian and Ubuntu, which select ext4 by
    default in the installer.  This is, frankly, a bad decision on their part.
    From its creation, ext4 was intended as a "stop-gap" to give people more
    time to migrate away from the ext* family of filesystems.  It shouldn't be
    used for fresh installs.

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
                assert!(
                    n <= rem,
                    "copy_file_range returned more bytes than requested"
                );
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
    /// Take a snapshot of the file and map it into memory.
    ///
    /// Note that changes to the snapshot will be discarded unless you call
    /// [`Mmap::commit`].
    ///
    /// # Performance
    ///
    /// If the filesystem _doesn't_ support reflinks (eg. ext4) then this
    /// will physically duplicate the file on disk.  If the file is large then
    /// clearly this will be slow and consume I/O bandwidth.  The duplicate will
    /// be deleted when the `Mmap` is dropped.
    ///
    /// If the filesystem _does_ support reflinks (XFS, btrfs) then we simply
    /// mark the file as "copy on write" until the `Mmap` is dropped.  This is
    /// O(1) and fast: on my machine it takes just 0.1 ms longer than a plain
    /// old `File::open()`.  Disk usage will not increase until the file is
    /// modified.
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

    /// Atomically replace the original file with the contents of the snapshot.
    ///
    /// # Performance
    ///
    /// It's a similar story to [`Mmap::open()`]: if the filesystem supports
    /// reflinks it'll be a fast O(1) (after waiting for writeback to finish);
    /// otherwise it'll be O(n).
    ///
    /// If you're done with the file you can use [`Mmap::commit_and_close`],
    /// which is always O(1).
    ///
    // # Semantics
    //
    // "Atomic commits" means: if you make two changes to a file and then
    // commit it, any future readers will either see both changes or neither of
    // them. The classic way to achieve this is by creating a new file with the
    // changes and linking it over the old one (ie. At the same path). Anyone
    // who subsequently opens the file will see the new contents. Any processes
    // that already had the file open will continue to see the old contents.
    // This technically satisfies the definition, although it is a shame you
    // have to re-open the file to see the changes. Linux offers syscalls
    // which let you atomically modify the contents of a file; no tricks, other
    // readers see the new contents without re-opening. Sadly these are not
    // available on all filesystems.  XFS and btrfs have support; ext4 does not
    // and probably never will.
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

    /// Atomically replace the original file with the contents of the snapshot and close it.
    ///
    /// Atomic and O(1).
    pub fn commit_and_close(mut self) -> io::Result<()> {
        match self.path.take() {
            Some(path) => self.link(path),
            None => self.commit(),
        }
    }

    /// Link this snapshot to the directory tree at the given path.
    ///
    /// Atomic and O(1) if `path` is on the same filesystem as the original
    /// file.
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
            // SAFETY: The pointer and length are valid as they were initialized in `open`
            // and are only invalidated in `drop`.
            unsafe {
                msync(self.ptr, self.len, MsyncFlags::SYNC)?;
            }
        }
        Ok(())
    }

    /// Change the size of the file.  If extending, the extension is filled with zeroes.
    pub fn resize(&mut self, new_len: usize) -> io::Result<()> {
        ftruncate(&self.private, new_len as u64)?;
        // SAFETY: `mremap` is safe here as it updates the mapping owned by `Mmap`.
        // The `MAYMOVE` flag allows the kernel to relocate the mapping.
        unsafe {
            self.ptr = mremap(self.ptr, self.len, new_len, MremapFlags::MAYMOVE)?;
        }
        self.len = new_len;
        Ok(())
    }
}

impl AsRef<[u8]> for Mmap {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: The memory region is valid and initialized for the duration of the slice.
        unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl AsMut<[u8]> for Mmap {
    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` ensures exclusive access in Rust, and the use of `O_TMPFILE`
        // ensures that no other process can modify the underlying file.
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: We are unmapping the region using the same pointer and length
        // that were used to create the mapping in `open`.
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
