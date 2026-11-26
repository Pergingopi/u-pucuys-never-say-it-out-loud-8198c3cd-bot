"""One-line confession entry for the "Never Say It Out Loud" bounty.

The program publishes a single fixed sentence to ``entry.txt``
inside the directory named by ``MAKEFUN_OUTPUT_DIR`` (``/output``
when the variable is unset, empty, or only whitespace). The same
sentence comes out of every run: no randomness, no clock, no
input, nothing to read from the network or the disk.

The sealed scorer only wants a UTF-8 file of 1 to 200 characters
that is not whitespace alone. This module holds itself to a
tighter bar: it refuses control characters, Unicode line and
paragraph separators, invisible format characters, every space
separator except the plain ASCII space (a non-breaking space is
refused), and any leading or trailing whitespace. What lands on
disk is exactly one clean line, nothing more.

Publication is done in stages, each checked before the next
begins. The payload is prepared and validated, written to a
staging file inside the output directory, flushed and fsynced,
read back and re-checked, then renamed into place atomically.
The directory itself is fsynced so the rename survives a crash,
and the published file is read back a second time together with
a check of its mode. Staging files abandoned by a crashed
earlier run are cleared first, but only files this program
could have made, and only once they are old enough that no live
run could own them.
"""

import os
import stat
import tempfile
import time
import unicodedata

ENTRY_LINE = (
    "I don't actually understand any of this \u2014 I just sound "
    "confident and hope you don't check."
)

FALLBACK_OUTPUT_DIR = "/output"
STAGING_PREFIX = ".entry-"
STAGING_SUFFIX = ".tmp"
# A live run holds its staging file for milliseconds, so anything
# this old can only be the leavings of a run that died mid-write.
# The age floor also protects a concurrent run: its fresh staging
# file is never old enough to be cleared by a neighbour.
ABANDONED_STAGING_MIN_AGE = 60.0
PUBLISHED_MODE = 0o644


def _output_directory():
    """Resolve the output directory from the environment.

    A missing variable, an empty value, or a value that is only
    whitespace all fall back to the default directory.
    """
    raw = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if raw is None:
        return FALLBACK_OUTPUT_DIR
    cleaned = raw.strip()
    return cleaned or FALLBACK_OUTPUT_DIR


def _is_forbidden_codepoint(ch):
    """True when one character may never appear in the entry.

    Forbidden means any control, format, surrogate, line-separator
    or paragraph-separator character (Unicode categories Cc, Cf,
    Cs, Zl, Zp), any space separator other than the plain ASCII
    space -- a non-breaking space and its cousins are refused --
    and every Unicode noncharacter: U+FDD0 through U+FDEF, plus
    any code point ending in FFFE or FFFF.
    """
    category = unicodedata.category(ch)
    if category in ("Cc", "Cf", "Cs", "Zl", "Zp"):
        return True
    if category == "Zs" and ch != " ":
        return True
    code = ord(ch)
    if 0xFDD0 <= code <= 0xFDEF:
        return True
    return (code & 0xFFFE) == 0xFFFE


def _entry_is_publishable(text):
    """Strict check that a candidate line may be published.

    The line must be a string of 1 to 200 characters, hold
    something besides whitespace, have no leading or trailing
    whitespace, encode cleanly as UTF-8 (a lone surrogate
    cannot), and contain no forbidden code point.
    """
    if not isinstance(text, str):
        return False
    if not 1 <= len(text) <= 200:
        return False
    if not text.strip():
        return False
    if text != text.strip():
        return False
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return not any(_is_forbidden_codepoint(ch) for ch in text)


def _clear_abandoned_staging_files(out_dir):
    """Delete staging files left behind by runs that crashed.

    Only files wearing this program's prefix and suffix are
    touched, and only once they pass the minimum age. Any error
    is swallowed: a failed sweep must never stop the run.
    Returns how many files were removed.
    """
    now = time.time()
    try:
        names = os.listdir(out_dir)
    except OSError:
        return 0
    removed = 0
    for name in names:
        if not (name.startswith(STAGING_PREFIX)
                and name.endswith(STAGING_SUFFIX)):
            continue
        path = os.path.join(out_dir, name)
        try:
            if now - os.path.getmtime(path) < ABANDONED_STAGING_MIN_AGE:
                continue
            os.unlink(path)
            removed += 1
        except OSError:
            continue
    return removed


def _decode_checked(raw_bytes, path):
    """Decode bytes as UTF-8, failing loudly on bad input."""
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise SystemExit("read-back is not valid UTF-8 in %s" % path)


def _verify_file_bytes(path, expected_bytes):
    """Read a file back and demand it match the payload exactly.

    The bytes must be identical, and the decoded text must pass
    the publish check. Returns the decoded text.
    """
    with open(path, "rb") as handle:
        back = handle.read()
    if back != expected_bytes:
        raise SystemExit("read-back mismatch in %s" % path)
    text = _decode_checked(back, path)
    if not _entry_is_publishable(text):
        raise SystemExit("read-back fails the gate check in %s" % path)
    return text


def _prepare_payload():
    """Validate the fixed entry line and encode it for writing.

    The line is a constant, so the payload is identical on every
    run; validation here is a guard against an edit that would
    quietly break the gate, not a branch that varies by run.
    """
    if not _entry_is_publishable(ENTRY_LINE):
        raise SystemExit("entry line fails the gate check")
    return ENTRY_LINE.encode("utf-8")


def _write_staging_file(out_dir, payload):
    """Write the payload to a staging file and fsync it.

    Returns the staging path. The file is created with mode 0644
    before and after the write so no umask can leave it wider.
    """
    fd, staging_path = tempfile.mkstemp(
        prefix=STAGING_PREFIX, suffix=STAGING_SUFFIX, dir=out_dir
    )
    # From here on the open descriptor belongs to the ``with``
    # block below; the except clause only runs before that
    # handover, when the descriptor is still ours to clean up.
    try:
        os.chmod(staging_path, PUBLISHED_MODE)
        with os.fdopen(fd, "wb") as handle:
            written = handle.write(payload)
            if written != len(payload):
                raise SystemExit(
                    "short write: %d of %d bytes" % (written, len(payload))
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staging_path, PUBLISHED_MODE)
    except BaseException:
        try:
            os.unlink(staging_path)
        except OSError:
            pass
        raise
    return staging_path


def _fsync_directory(out_dir):
    """Fsync the output directory so the rename is durable."""
    dir_fd = os.open(out_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _publish_atomically(out_dir, payload):
    """Stage, verify, rename into place, and verify the target."""
    _clear_abandoned_staging_files(out_dir)
    staging_path = _write_staging_file(out_dir, payload)
    try:
        # Verify the staged copy before it becomes the real file.
        _verify_file_bytes(staging_path, payload)

        target = os.path.join(out_dir, "entry.txt")
        os.replace(staging_path, target)
        os.chmod(target, PUBLISHED_MODE)
        _fsync_directory(out_dir)

        # Verify the published file itself, and its mode.
        _verify_file_bytes(target, payload)
        mode = stat.S_IMODE(os.stat(target).st_mode)
        if mode != PUBLISHED_MODE:
            raise SystemExit("output mode is not 0644: %o" % mode)
    finally:
        if os.path.exists(staging_path):
            os.unlink(staging_path)


def main():
    out_dir = _output_directory()
    os.makedirs(out_dir, exist_ok=True)
    payload = _prepare_payload()
    _publish_atomically(out_dir, payload)


if __name__ == "__main__":
    main()
