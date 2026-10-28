# #0012 Read-only board inventory helper failed: Python 3.9 Path.stat rejects follow_symlinks; use Path.lstat for metadata.

- 2026-10-28T11:08:15Z `issue`: Read-only board inventory helper failed: Python 3.9 Path.stat rejects follow_symlinks; use Path.lstat for metadata. [fpga read-only storage inventory]
- 2026-10-28T11:08:41Z `attempt`: Changed the inline read-only inventory helper to Path.lstat; it completed on board Python 3.9 and returned model/media metadata, archive sizes and output-file counts. No files changed on the board. [fpga read-only storage inventory] (worked)
- 2026-10-28T11:08:46Z `fix`: Inline inventory succeeds using Path.lstat instead of Path.stat(follow_symlinks=False), unsupported by board Python 3.9. This was an inspection-command compatibility error, not a repository defect. [fpga read-only storage inventory]
