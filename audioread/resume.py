# This file is part of audioread.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

"""Resumable batch decoding with persistent, atomically-updated state.

This module is strictly opt-in: it adds a checkpoint layer *on top of*
the existing per-file decoding API without changing it. Callers that
never touch this module observe no behavioral difference.

Key properties:

- State is persisted as JSON and updated atomically (write to a
  temporary file, fsync, then ``os.replace``). A ``kill -9`` can never
  leave a half-written state file.
- Decoded output files are written to a temporary sibling and renamed
  into place only after being fully written, so a truncated output is
  never mistaken for a finished one.
- Completed items are validated on resume (size + SHA-256 checksum);
  anything that does not match is decoded again.
- A lock file (``fcntl.flock``) prevents two processes from running
  the same batch concurrently.
- The state format is versioned and tolerant: unknown fields are
  ignored so older code can read newer state files.
"""

import contextlib
import errno
import fcntl
import hashlib
import json
import logging
import os
import tempfile

from .exceptions import DecodeError

log = logging.getLogger(__name__)

# Version of the on-disk state format. Readers must ignore unknown
# fields so that older code keeps working with newer state files.
STATE_VERSION = 1

_STATE_SUFFIX = '.state.json'
_LOCK_SUFFIX = '.lock'


class ResumeError(Exception):
    """Base class for errors raised by the resume/checkpoint layer."""


class StateCorruptedError(ResumeError):
    """The state file exists but cannot be parsed or is inconsistent.

    Raised instead of silently restarting the batch, which would redo
    work and potentially duplicate results.
    """


class BatchConflictError(ResumeError):
    """Another process is currently running this batch."""


class BatchResult:
    """Summary of a batch run.

    `completed` lists input paths finished in this run, `skipped`
    lists inputs already done according to a previous run's state, and
    `failed` maps input paths to their error messages.
    """

    def __init__(self):
        self.completed = []
        self.skipped = []
        self.failed = {}

    @property
    def ok(self):
        """True when every item is either completed or skipped."""
        return not self.failed

    def __repr__(self):
        return ('BatchResult(completed=%d, skipped=%d, failed=%d)' %
                (len(self.completed), len(self.skipped),
                 len(self.failed)))


def _default_output_path(input_path, output_dir):
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, base + '.raw')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path, data):
    """Write `data` as JSON to `path` atomically.

    The JSON is written to a temporary file in the same directory,
    flushed and fsynced, then renamed over `path`. If anything fails
    (including a full disk), the pre-existing state file is left
    untouched and the temporary file is removed.
    """
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix='.tmp-state-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        if exc.errno == errno.ENOSPC:
            log.error('resume: no space left while writing state file '
                      '%s; previous state preserved', path)
        raise


def _load_state(state_path):
    """Load and validate the state file, tolerating unknown fields.

    Returns a dict with at least the keys `version` and `items`.
    Raises StateCorruptedError if the file cannot be parsed or its
    structure is invalid.
    """
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            raw = f.read()
    except OSError as exc:
        raise StateCorruptedError(
            'cannot read state file %s: %s' % (state_path, exc))

    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise StateCorruptedError(
            'state file %s is corrupted (invalid JSON: %s); fix or '
            'remove it manually -- refusing to silently restart the '
            'batch' % (state_path, exc))

    if not isinstance(data, dict) or not isinstance(
            data.get('items'), dict):
        raise StateCorruptedError(
            'state file %s has an invalid structure; fix or remove it '
            'manually -- refusing to silently restart the batch'
            % state_path)

    version = data.get('version', STATE_VERSION)
    if not isinstance(version, int):
        raise StateCorruptedError(
            'state file %s has an invalid version field' % state_path)
    if version > STATE_VERSION:
        # Forward compatibility: keep the fields we understand and
        # ignore the rest instead of crashing.
        log.warning('resume: state file %s has version %d, newer than '
                    'supported version %d; unknown fields will be '
                    'ignored', state_path, version, STATE_VERSION)

    items = {}
    for key, record in data['items'].items():
        if not isinstance(record, dict):
            raise StateCorruptedError(
                'state file %s has an invalid record for item %r'
                % (state_path, key))
        # Keep only the fields this version understands; drop anything
        # written by a newer version.
        items[key] = {
            'status': record.get('status'),
            'output': record.get('output'),
            'size': record.get('size'),
            'sha256': record.get('sha256'),
            'error': record.get('error'),
        }
    return {'version': min(version, STATE_VERSION), 'items': items}


def _new_state():
    return {'version': STATE_VERSION, 'items': {}}


class _BatchLock:
    """Advisory lock guarding a batch's state file.

    Uses fcntl.flock on a sibling lock file so that a process killed
    with SIGKILL releases the lock automatically.
    """

    def __init__(self, lock_path):
        self._lock_path = lock_path
        self._file = None

    def __enter__(self):
        self._file = open(self._lock_path, 'a')
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._file.close()
            self._file = None
            raise BatchConflictError(
                'batch state %s is locked by another process; refusing '
                'to run the same batch concurrently' % self._lock_path)
        return self

    def __exit__(self, *exc_info):
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None
        return False


def _decode_to_file(input_path, output_path, backends=None):
    """Decode one audio file to raw s16le PCM, atomically.

    The audio is decoded with the existing backend machinery and
    written to a temporary sibling file that is renamed over
    `output_path` only once decoding has finished. Returns a
    (size, sha256) tuple describing the final output.
    """
    from . import audio_open

    directory = os.path.dirname(os.path.abspath(output_path))
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix='.tmp-decode-', suffix='.raw')
    try:
        with os.fdopen(fd, 'wb') as out:
            with audio_open(input_path, backends=backends) as f:
                for buf in f:
                    out.write(buf)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, output_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return os.path.getsize(output_path), _sha256(output_path)


def _cleanup_stale_temps(*directories):
    """Remove leftover temp files from a previously killed run.

    Only files with our own prefixes are touched. Called while holding
    the batch lock, so no live process can own these files.
    """
    for directory in directories:
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            if name.startswith(('.tmp-state-', '.tmp-decode-')):
                log.warning('resume: removing stale temp file %s',
                            os.path.join(directory, name))
                with contextlib.suppress(OSError):
                    os.unlink(os.path.join(directory, name))


def _item_done(record):
    """Check whether a state record still matches its output file."""
    if record.get('status') != 'done':
        return False
    output = record.get('output')
    size = record.get('size')
    checksum = record.get('sha256')
    if not output or size is None or checksum is None:
        return False
    if not os.path.exists(output):
        return False
    if os.path.getsize(output) != size:
        log.warning('resume: %s has unexpected size, reprocessing',
                    output)
        return False
    if _sha256(output) != checksum:
        log.warning('resume: %s has unexpected checksum, reprocessing',
                    output)
        return False
    return True


def decode_batch(inputs, output_dir, state_path=None, backends=None,
                 retry_failed=True, output_path=None):
    """Decode a batch of audio files with checkpoint/resume support.

    Each input is decoded to raw s16le PCM (interleaved, using the
    source channel count and sample rate) inside `output_dir`. Progress
    is recorded in `state_path` (default: ``<output_dir>/.batch`` plus
    a state suffix) after every item, atomically.

    On a subsequent run, items whose recorded output still matches its
    size and checksum are skipped; everything else -- including items
    previously marked failed when `retry_failed` is true -- is decoded
    again. Running the same batch twice is idempotent.

    `output_path` may be a callable mapping an input path to its output
    path; by default ``<output_dir>/<basename>.raw`` is used.

    Returns a BatchResult. Individual decode failures are recorded in
    the state file and in `BatchResult.failed` instead of raising, so
    one bad file does not abort the batch. Errors in the checkpoint
    machinery itself (corrupted state, lock conflicts, I/O errors such
    as a full disk) are raised as ResumeError/OSError.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if state_path is None:
        state_path = os.path.join(output_dir, '.batch' + _STATE_SUFFIX)
    lock_path = state_path + _LOCK_SUFFIX

    log.debug('resume: entering batch of %d item(s), state=%s',
              len(inputs), state_path)

    with _BatchLock(lock_path):
        _cleanup_stale_temps(os.path.dirname(state_path), output_dir)
        if os.path.exists(state_path):
            state = _load_state(state_path)
        else:
            state = _new_state()

        result = BatchResult()
        for input_path in inputs:
            input_path = os.path.abspath(input_path)
            if output_path is not None:
                out_path = output_path(input_path)
            else:
                out_path = _default_output_path(input_path, output_dir)

            record = state['items'].get(input_path)
            if record is not None and _item_done(record):
                log.debug('resume: skipping completed item %s',
                          input_path)
                result.skipped.append(input_path)
                continue
            if (record is not None and record.get('status') == 'failed'
                    and not retry_failed):
                result.failed[input_path] = record.get('error') or \
                    'previous failure'
                continue

            try:
                size, checksum = _decode_to_file(input_path, out_path,
                                                 backends=backends)
            except (DecodeError, OSError) as exc:
                log.warning('resume: failed to decode %s: %s',
                            input_path, exc)
                state['items'][input_path] = {
                    'status': 'failed',
                    'output': out_path,
                    'size': None,
                    'sha256': None,
                    'error': str(exc),
                }
                _atomic_write_json(state_path, state)
                result.failed[input_path] = str(exc)
                continue

            state['items'][input_path] = {
                'status': 'done',
                'output': out_path,
                'size': size,
                'sha256': checksum,
                'error': None,
            }
            _atomic_write_json(state_path, state)
            result.completed.append(input_path)

    if result.failed:
        log.warning('resume: batch finished with %d failure(s)',
                    len(result.failed))
    else:
        log.debug('resume: batch finished, %d completed, %d skipped',
                  len(result.completed), len(result.skipped))
    return result
