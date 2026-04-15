#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]

/*! **Safe** `mmap()` with **snapshot isolation** and **atomic commits**.

([Linux-only](#os-support), [works best](#performance) on XFS/btrfs.)

## Example

```rust
# use mmap_snapshot::Mmap;
# fn foo() -> std::io::Result<()> {
# let path = std::path::Path::new("/tmp/foo");
# std::fs::write(&path, b"Hello world!")?;
let mut mmap = Mmap::open(&path)?;
assert_eq!(mmap.len(), 12);
assert_eq!(&mmap[..], b"Hello world!");
mmap[6..11].copy_from_slice(b"sekai");
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
    ops::{Deref, DerefMut},
    os::fd::AsFd,
    path::{Path, PathBuf},
};

/// Returns whether it fell back
fn ficlone(fd_out: impl AsFd, fd_in: impl AsFd, len: usize) -> io::Result<bool> {
    match ioctl_ficlone(&fd_out, &fd_in) {
        Ok(()) => Ok(false),
        Err(Errno::OPNOTSUPP) => {
            ftruncate(&fd_out, len as u64)?;
            let mut off_in = 0;
            let mut off_out = 0;
            while off_in < len as u64 {
                let rem = len - off_in as usize;
                let n =
                    copy_file_range(&fd_in, Some(&mut off_in), &fd_out, Some(&mut off_out), rem)?;
                assert_eq!(off_in, off_out);
                assert!(
                    n <= rem,
                    "copy_file_range() copied more bytes than requested"
                );
                if n == 0 {
                    Err(io::ErrorKind::UnexpectedEof)?;
                }
            }
            assert_eq!(off_out, len as u64);
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
/// Read the file contents using the `Deref` impl.  The data you see will
/// reflect the state of the file at the time `open()` was called; writes by other
/// process are not reflected.  In other words, `Mmap` will show you a consistent
/// point-in-time snapshot of the file.
///
/// Data is not loaded eagerly into memory.  It will be read in from disk on demand.
/// For this we rely on the COW capabilities of the underlying filesystem.
///
/// ## Writing
///
/// Modify the contents using the `DerefMut` impl.  Writes will not be visible
/// to other processes reading the file until you call `commit()`.  Once you
/// call `commit()`, all your modifications will be atomically visible to other
/// readers.  If you drop the `Mmap` without calling `commit()`, your writes
/// will be lost!
///
/// Modifications are written to disk continuously in the background; `commit()`
/// simply waits for writeback to finish, and then makes the written changes
/// visible.
pub struct Mmap {
    original: OriginalFile,
    private: File, // Unlinked; initially a clone of `original`
    ptr: *mut c_void,
    len: usize,
}

enum OriginalFile {
    /// In this case the file is on a reflink-capable filesystem
    Fd(File),
    /// In this case the file is on a reflink-incapable filesystem
    Path(PathBuf),
}

// SAFETY:
// All members of `Mmap` implement Send except for `ptr`.  `ptr`+`len` are just
// "plain old memory" (that's the point of the trick with the private unlinked
// file.)  So they have the same properties as Box<[u8]> wrt Send/Sync.
unsafe impl Send for Mmap {}
// SAFETY: See above
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
        if len >= isize::MAX as usize {
            return Err(io::ErrorKind::FileTooLarge.into());
        }
        let dir = path.parent().unwrap_or(Path::new("."));
        let private: File =
            open(dir, OFlags::TMPFILE | OFlags::RDWR, Mode::RUSR | Mode::WUSR)?.into();
        let fellback = ficlone(&private, &original, len)?;

        let ptr;
        if len == 0 {
            ptr = std::ptr::null_mut();
        } else {
            // SAFETY:
            // > If `ptr` is not null, it must be aligned...
            //
            // `ptr` is null.
            //
            // > If there exist any Rust references referring to the memory region
            //
            // We're letting the kernel pick an unused region so there shouldn't be any.
            //
            // > or if you subsequently create a Rust reference referring to the
            // > resulting region,
            //
            // We will be doing this.
            //
            // > it is your responsibility to ensure that the Rust reference invariants are
            // > preserved, including ensuring that the memory is not mutated in a way that
            // > a Rust reference would not expect.
            //
            // See the safety comment in the DerefMut impl.
            unsafe {
                ptr = mmap(
                    std::ptr::null_mut(),
                    len,
                    ProtFlags::READ | ProtFlags::WRITE,
                    MapFlags::SHARED,
                    &private,
                    0,
                )?;
            }
        };
        assert!(ptr.is_null() == (len == 0));
        Ok(Self {
            private,
            ptr,
            len,
            original: if fellback {
                OriginalFile::Path(path.to_owned())
            } else {
                OriginalFile::Fd(original)
            },
        })
    }

    /// Atomically replace the original file with the contents of the snapshot.
    ///
    /// You can continue to read/write the mmap after calling `commit()`.
    ///
    /// # Performance
    ///
    /// It's a similar story to [`Mmap::open()`]: if the filesystem supports
    /// reflinks it'll be a fast O(1) (after waiting for writeback to finish);
    /// otherwise it'll be O(n).
    ///
    /// If you're done with the file you can use [`Mmap::commit_and_close`],
    /// which is always O(1).
    pub fn commit(&mut self) -> io::Result<()> {
        self.sync()?;
        match &self.original {
            OriginalFile::Fd(original) => ioctl_ficlone(original, &self.private)?,
            OriginalFile::Path(path) => {
                // We can't just copy self.private to self.original, since
                // this would not be atomic. And we need to keep self.private
                // unlinked. So we create a new private file, copy over the
                // contents, and link it.
                let dir = path.parent().unwrap_or(Path::new("."));
                let private2: File =
                    open(dir, OFlags::TMPFILE | OFlags::RDWR, Mode::RUSR | Mode::WUSR)?.into();
                // This is non-atomic but that's fine, since we're holding &mut
                // self and therefore `self.private` can't receive modifications
                // while the copy is in-progress
                ficlone(&private2, &self.private, self.len)?;
                linkat(&private2, "", rustix::fs::CWD, path, AtFlags::EMPTY_PATH)?;
            }
        }
        Ok(())
    }

    /// Atomically replace the original file with the contents of the snapshot and close it.
    ///
    /// Atomic and O(1).
    pub fn commit_and_close(mut self) -> io::Result<()> {
        match &self.original {
            OriginalFile::Fd(_) => self.commit(),
            OriginalFile::Path(path) => {
                let path = path.clone();
                // `path` is always on the same filesystem as the original file - it
                // _is_ the original file!  So this is atomic.
                self.link(path)
            }
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
            // SAFETY:
            // > `addr` must be a valid pointer to memory that is appropriate to call
            // > `msync` on.
            //
            // Given that len is non-zero, `self.ptr` is a pointer which
            // came from `mmap()`, and `self.len` is the length we passed to
            // `mmap()`, so together these describe an mmapped region and are
            // safe to pass to `msync()`.
            unsafe {
                msync(self.ptr, self.len, MsyncFlags::SYNC)?;
            }
        }
        Ok(())
    }

    /// Change the size of the file.  If extending, the extension is filled with zeroes.
    pub fn resize(&mut self, new_len: usize) -> io::Result<()> {
        if new_len >= isize::MAX as usize {
            return Err(io::ErrorKind::FileTooLarge.into());
        }
        if new_len == self.len {
            return Ok(());
        }
        ftruncate(&self.private, new_len as u64)?;
        if new_len == 0 {
            // SAFETY: See the Drop impl
            unsafe {
                munmap(self.ptr, self.len)?;
            }
            self.ptr = std::ptr::null_mut();
        } else if self.len == 0 {
            // SAFETY: See Mmap::open()
            unsafe {
                self.ptr = mmap(
                    std::ptr::null_mut(),
                    new_len,
                    ProtFlags::READ | ProtFlags::WRITE,
                    MapFlags::SHARED,
                    &self.private,
                    0,
                )?;
            }
        } else {
            // SAFETY:
            // > `self.ptr` must be aligned to the applicable page size, and the range of
            // > memory starting at `self.ptr` and extending for `self.len` bytes,
            // > rounded up to the applicable page size, must be valid to mutate with
            // > `self.ptr`'s provenance.
            //
            // `self.ptr` comes from an mmap with length `self.len`, so this should
            // all hold.
            //
            // > If `MremapFlags::MAY_MOVE` is set in `flags`,
            // > there must be no Rust references referring to that the memory.
            //
            // This flag is set, so `mremap()` might move the mapping to a
            // completely new address. The only way to get references to the mapping
            // is via the Deref/DerefMut impls, which take borrows on the `Mmap`. Since
            // this method takes `&mut self`, we know that no such references are
            // live.
            //
            // > If `new_len` is less than `self.len`, than there must be no Rust
            // > references referring to the memory starting at offset `new_len` and ending
            // > at `self.len`.
            //
            // As per the above, there are no live references at all into the
            // mapping.
            unsafe {
                self.ptr = mremap(self.ptr, self.len, new_len, MremapFlags::MAYMOVE)?;
            }
        }
        self.len = new_len;
        assert!(self.ptr.is_null() == (self.len == 0));
        Ok(())
    }
}

impl Deref for Mmap {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: See the `DerefMut` impl.
            unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }
}

impl DerefMut for Mmap {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY:
            // > `ptr` must be valid for both reads and writes for `len *
            // >  size_of::<T>()` many bytes ...
            // > The entire memory range of this slice must be contained within a
            // >  single allocation!
            //
            // The whole range comes from a single call to `mmap()` with length
            // `len`.
            //
            // > `ptr` must be non-null
            //
            // So long as len is non-zero, `ptr` is asserted to be non-null wherever
            // it is modified (`open()` and `resize()`).
            //
            // > `ptr` must be properly aligned
            //
            // The element type is `u8`, so `ptr` is trivially aligned.
            //
            // > `ptr` must point to `len` consecutive properly initialized values
            // > of type `u8`.
            //
            // File-backed VMAs count as initialized.  There's no such thing as a
            // file which contains uninitialised bytes.  (Even sparse regions are
            // well-defined as containing zeroes.)
            //
            // > The memory referenced by the returned slice must not be accessed
            // > through any other pointer (not derived from the return value) for
            // > the duration of lifetime `'a`. Both read and write accesses are
            // > forbidden.
            //
            // This is the big one.  I believe this is satisfied if both of the
            // following hold true:
            //
            // * the only way to mutate the memory is via this `DerefMut` impl
            // * the only way to read the memory is via this `DerefMut` impl or the
            //   `Deref` impl
            //
            // This memory can be accessed via these impls of course, and also via
            // operations on the underlying file. However, we can be sure that no
            // such file operations will take place. That's because:
            //
            // * The file was created with `O_TMPFILE`, which means it's impossible
            //   to create a new fd for the file via the filesystem.
            // * We never expose our fd, which means it's impossible to create a new
            //   fd via `clone()`.
            // * Therefore the _only_ fd referencing the underlying file is
            //   `self.private`.
            // * All public methods which access the fd (self.private) take
            //   `&mut self`.
            // * Therefore we don't access the file via that fd while `'a` is live.
            // * Therefore the memory can only be accessed via the mmap
            //
            // > The total size `len * size_of::<T>()` of the slice must be no
            // > larger than `isize::MAX`, and adding that size to `ptr` must not
            // > "wrap around" the address space. See the safety documentation of
            // > [`pointer::offset`].
            //
            // `mmap()` puts the mapping somewhere where it fits, so
            // `self.ptr.add(self.len)` will never overflow the address space.
            // `self.len < isize::MAX` is asserted in open() and resize().
            unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY:
            // > `ptr` must be aligned to the applicable page size, and the range of memory
            // > starting at `ptr` and extending for `len` bytes, rounded up to the
            // > applicable page size, must be valid to mutate with `ptr`'s provenance.
            //
            // `self.ptr` comes from an mmap with length `self.len`.
            //
            // > And there must be no Rust references referring to that memory.
            //
            // The only way to get references to the mapping is via the
            // Deref/DerefMut impls, which take borrows on the `Mmap`.  Since this
            // method takes `&mut self`, we know that no such references are live.
            unsafe {
                match munmap(self.ptr, self.len) {
                    Ok(()) => (),
                    Err(e) => eprintln!("munmap failed: {e}"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: These could use some improvement.  Ideally I'd mount a bunch
    // of different filesystems... but that requires root.  Anyway, I should
    // systematically make sure the files used by different tests don't
    // conflict, and that they're cleaned up at the end.

    fn paths(name: &str) -> impl Iterator<Item = PathBuf> {
        ["/tmp", "/var/tmp"].into_iter().map(move |d| {
            let d = Path::new(d).join("mmap-snapshot");
            std::fs::create_dir_all(&d).unwrap();
            d.join(name)
        })
    }

    #[test]
    fn mmap() -> std::io::Result<()> {
        for p in paths("mmap") {
            std::fs::write(&p, b"Hello world!")?;
            let f = Mmap::open(&p)?;
            std::fs::write(&p, b"Goodbye world!")?;
            assert_eq!(&*f, b"Hello world!");
            std::fs::remove_file(&p)?;
            assert_eq!(&*f, b"Hello world!");
        }
        Ok(())
    }

    #[test]
    fn mmap_mut() -> std::io::Result<()> {
        for p in paths("mmap_mut") {
            std::fs::write(&p, b"Hello world!")?;
            let mut f = Mmap::open(&p)?;
            assert_eq!(&*f, b"Hello world!");
            f[6..11].copy_from_slice(b"sekai");
            assert_eq!(&*f, b"Hello sekai!");
            assert_eq!(std::fs::read_to_string(&p)?, "Hello world!");
            f.commit()?;
            std::mem::drop(f);
            assert_eq!(std::fs::read_to_string(&p)?, "Hello sekai!");
            std::fs::remove_file(&p)?;
        }
        Ok(())
    }

    #[test]
    fn zero_len() -> std::io::Result<()> {
        for p in paths("zero_len") {
            File::create(&p)?;
            let f = Mmap::open(&p)?;
            assert_eq!(&*f, b"");
            std::fs::remove_file(&p)?;
            assert_eq!(&*f, b"");
        }
        Ok(())
    }

    #[test]
    fn zero_len_mut() -> std::io::Result<()> {
        for p in paths("zero_len_mut") {
            File::create(&p)?;
            let mut f = Mmap::open(&p)?;
            assert_eq!(&*f, b"");
            f.resize(12)?;
            f.copy_from_slice(b"Hello world!");
            assert_eq!(std::fs::read_to_string(&p)?, "");
            f.commit()?;
            assert_eq!(std::fs::read_to_string(&p)?, "Hello world!");
            f[6..11].copy_from_slice(b"sekai");
            assert_eq!(&*f, b"Hello sekai!");
            assert_eq!(std::fs::read_to_string(&p)?, "Hello world!");
            f.commit()?;
            std::mem::drop(f);
            assert_eq!(std::fs::read_to_string(&p)?, "Hello sekai!");
            std::fs::remove_file(&p)?;
        }
        Ok(())
    }
}
