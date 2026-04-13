<h1 align="center">mmap-snapshot</h1>

Safe `mmap()` for Rust, with snapshot isolation and atomic commits.
(Linux-only, works best on XFS/btrfs.)

## The better interface for file I/O

Ah, the classic UNIX I/O interface: `read()`, `write()`, `seek()`.
It's the perfect interface for operating a magnetic tape.

But these days, whether it's magnetic tape, spinning rust, or NAND,
you're not really operating the device any more.
All file I/O is mediated by the page cache[^direct],
which means these `read()`/`write()` are not actually performing I/O at all -
they're performing memcpys.
The actual I/O happens when synchronising the page cache and the disk.
This _could_ happen during the syscall,
or it might be deferred until later,
or it might have already happened.

The natural interface for interacting with the page cache is `mmap()`.
This syscall maps the page cache into your address space
so you can operate on it directly.
Copying bytes from the page cache to a buffer and back again is an unnecessary dance
which complicates your code and makes it slower.




So what's the catch?
There are a couple (see [below](#other-footguns)),
but let's start with the big one:
Other processes can modify the file while you have it mapped.
When that happens, the bytes in your address space are updated to reflect the changes[^private].
All processes which mmap the file share the same memory.
The upshot: what you get is less `&[u8]` and more `&[AtomicU8]` -
not looking so convenient any more!
This is why `mmap()`s which return a `&[u8]` generally require `unsafe`
and a solemn promise that the file is immutable.

[^direct]: You can skip the page cache using direct I/O, but approximately no-one does that
[^private]: And setting `MAP_PRIVATE` actually makes it _worse_:
  changes might not propogate to your mapping for a while, but they will eventually.
  The result is your mysterious UB is doubly mysterious.

## Private files can be mmapped safely

In Linux, you can create a file without linking it to the directory tree (ie. a file with no name).
No path to the file means other processes can’t open it[^proc],
and that means the only process which can modify the file is the one that created it.
If we use the fd to create an mmap _and nothing else_
then all modifications to the file must be made via the mmap.[^io_safe]
This solves the "spontaneous mutation" problem
and allows us to safely present it as `&mut [u8]`.
Read more about [O_TMPFILE].

[^proc]: Unless they go snooping in /proc.  But such adventures aren't covered by Rust's safety guarantees.
[^io_safe]: You might wonder about code in your own process `write()`ing to
  random fds.  Such behaviour is also [not covered][io safety] by Rust's safety
  guarantees: the tmpfile is an "exclusively owned fd", which "no other code is
  allowed to access in any way".

[O_TMPFILE]: https://man7.org/linux/man-pages/man2/open.2.html#:~:text=O%5FTMPFILE,-%28since
[io safety]: https://doc.rust-lang.org/nightly/std/io/index.html#io-safety

But typically we want to mmap an existing file, which is already linked to the directory tree.
Can we do that safely?
Yes: we create an unlinked _copy_ of the file and mmap that.
Then, when we're done, we replace the original file with the copy.

## Side-effect: everything is atomic now!

The data in the mmap reflects the state of the file at the instant we cloned it,
regardless of when it's faulted into memory.
Think "snapshot isolation".

Changes to the mmap are written to disk eagerly but aren't visible to other readers of the file
until you "commit" them (link the private clone back over the original file.)

## Clones are cheap (...on participating filesystems)

You might be thinking "copy the file..!?".
Well I have good news:
if your filesystem supports a feature called "reflinks" then
you can cheaply clone a file without actually copying any data.
On my (XFS-based) system it takes 0.1 ms to clone a file like this, regardless of its size.
Read more about [FICLONE].

That means the total cost of pulling this trick is 0.1 ms added to the initial setup,
which is small compared to the other setup costs (opening the file and mmapping it).
That's an easy sell!
..._if_ the filesystem supports reflinks.
Therein lies the rub:
if the file happens to be on a reflink-less filesystem then
we have to fallback to actually copying it, which is O(size).
Ouch, potentially.

Most new installs probably use XFS or btrfs,
but of course ext4 is common and will remain so for many years to come.
The conservative (Debian-based) distros are still defaulting to ext4 as of this writing!
But if you know that your code will be running exclusively on modern filesystems then you don't need to worry.

[FICLONE]: https://man7.org/linux/man-pages/man2/FICLONE.2const.html

## Putting it together

You could write this:

```rust
let mut data = std::fs::read("foo.dat")?;
data.reverse();
std::fs::write("foo.dat", data)?;
```

(Runs in ...ms)

Or this:

```rust
let mut buf = vec![0; 4096];
let mut file = File::open("foo.dat");
loop {
    let n = file.read(&mut buf)?;
    if n == 0 { break; }
    buf[..n].reverse();
    file.seek(SeekFrom::Current(-n))?;
    file.write(&buf[..n])?;
}
```

(Runs in ...ms)

Instead, write this:

```rust
let mut data = Mmap::open("foo.dat")?;
data.reverse();
data.commit()?;
```

(Runs in ...ms)

TODO: Take measurements for the 3 above.

## Other footguns

mmap is great but it's not perfect.  Here are the drawbacks I'm aware of:

1. I/O errors are reported via a signal.
  A byte in the page cache actually has 257 possible states: 0-255, and "poison".
  "Poison" means there was a hardware problem and this region of the file is corrupt.
  `read()` checks for poison before copying and returns a nice error.
  With `mmap()` you get hit with SIGBUS.

  If your game plan for dealing with I/O errors was going to be to panic anyway, then this isn't _much_ of a downgrade.
  If you were planning to actually handle this kind of error then SIGBUS will make your life harder.
1. Unpredictable latency.
  This might be a turn-off for `async` users.
1. If someone truncates the file and you read beyond the new EOF you get SIGBUS.
  This crate solves this issue, however.

And here are some drawbacks of the "private clone" trick:

1. Writes increase disk usage until committed.
  This is unavoidable if you want atomic commits.
1. Reflinks increase file fragmentation.
  This... might be avoidable with more cleverness.

## Prior art

Immutable files can be safely mmapped without this trick.  That means files which:

* come from erofs/squashfs
* have fsverity enabled
* are a fully-sealed memfd

See [safe-mmap] for a crate that supports this use-case.

[safe-mmap]: https://crates.io/crates/safe-mmap
