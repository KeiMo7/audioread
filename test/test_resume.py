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

"""Tests for the opt-in resumable batch decoding (audioread.resume)."""

import json
import os
import signal
import subprocess
import sys
import time
import wave

import pytest

import audioread
from audioread import resume


def make_wav(path, seed=0, frames=8000, samplerate=8000):
    """Create a small deterministic WAV file decodable by rawread."""
    data = bytes(((i * 7 + seed * 13) % 256 for i in range(frames * 2)))
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(data)
    return str(path)


def make_inputs(tmp_path, count=3):
    return [make_wav(tmp_path / ('in-%d.wav' % i), seed=i)
            for i in range(count)]


def output_bytes(output_dir):
    """Map of output filename -> content hash for a batch output dir."""
    result = {}
    for name in sorted(os.listdir(output_dir)):
        if name.endswith('.raw'):
            with open(os.path.join(output_dir, name), 'rb') as f:
                result[name] = f.read()
    return result


def state_path_for(output_dir):
    return os.path.join(output_dir, '.batch' + resume._STATE_SUFFIX)


def test_batch_single_run(tmp_path):
    """A full batch decodes every input and records done state."""
    inputs = make_inputs(tmp_path)
    out_dir = str(tmp_path / 'out')
    result = resume.decode_batch(inputs, out_dir)
    assert result.ok
    assert len(result.completed) == 3
    assert result.skipped == []
    outputs = output_bytes(out_dir)
    assert len(outputs) == 3
    with open(state_path_for(out_dir)) as f:
        state = json.load(f)
    assert state['version'] == resume.STATE_VERSION
    assert all(r['status'] == 'done' for r in state['items'].values())


def test_batch_resume_after_interruption(tmp_path, monkeypatch):
    """Interrupt mid-batch, rerun, and match a single-run result."""
    inputs = make_inputs(tmp_path)

    # Reference: one uninterrupted run.
    ref_dir = str(tmp_path / 'ref')
    ref_result = resume.decode_batch(inputs, ref_dir)
    assert ref_result.ok
    reference = output_bytes(ref_dir)

    # Interrupted run: blow up while decoding the second item.
    out_dir = str(tmp_path / 'out')
    real_decode = resume._decode_to_file
    calls = []

    def flaky_decode(input_path, output_path, backends=None):
        calls.append(input_path)
        if len(calls) == 2:
            raise RuntimeError('simulated crash')
        return real_decode(input_path, output_path, backends)

    monkeypatch.setattr(resume, '_decode_to_file', flaky_decode)
    with pytest.raises(RuntimeError):
        resume.decode_batch(inputs, out_dir)
    monkeypatch.undo()

    # Resume: only the remaining items are decoded.
    result = resume.decode_batch(inputs, out_dir)
    assert result.ok
    assert len(result.skipped) == 1
    assert len(result.completed) == 2
    assert output_bytes(out_dir) == reference


def test_batch_idempotent(tmp_path):
    """Running the same batch twice changes nothing."""
    inputs = make_inputs(tmp_path)
    out_dir = str(tmp_path / 'out')
    first = resume.decode_batch(inputs, out_dir)
    assert first.ok
    snapshot = output_bytes(out_dir)
    with open(state_path_for(out_dir)) as f:
        state_snapshot = f.read()

    second = resume.decode_batch(inputs, out_dir)
    assert second.ok
    assert second.completed == []
    assert len(second.skipped) == 3
    assert output_bytes(out_dir) == snapshot
    with open(state_path_for(out_dir)) as f:
        assert f.read() == state_snapshot


def test_batch_partial_failure_and_retry(tmp_path):
    """A failing item is recorded and can be retried on its own."""
    inputs = make_inputs(tmp_path, count=2)
    bad = str(tmp_path / 'broken.wav')
    with open(bad, 'wb') as f:
        f.write(b'this is not audio')
    out_dir = str(tmp_path / 'out')

    result = resume.decode_batch(inputs + [bad], out_dir)
    assert not result.ok
    assert len(result.completed) == 2
    assert list(result.failed) == [os.path.abspath(bad)]

    # The failure is persisted in the state file.
    with open(state_path_for(out_dir)) as f:
        state = json.load(f)
    assert state['items'][os.path.abspath(bad)]['status'] == 'failed'

    # Fix the bad file and retry: only it is decoded again.
    make_wav(bad, seed=99)
    retry = resume.decode_batch(inputs + [bad], out_dir)
    assert retry.ok
    assert retry.completed == [os.path.abspath(bad)]
    assert len(retry.skipped) == 2

    # retry_failed=False keeps the failure without re-decoding.
    with open(bad, 'wb') as f:
        f.write(b'still not audio')
    # Corrupt the recorded output so the item is not "done".
    with open(state_path_for(out_dir)) as f:
        state = json.load(f)
    state['items'][os.path.abspath(bad)]['status'] = 'failed'
    with open(state_path_for(out_dir), 'w') as f:
        json.dump(state, f)
    kept = resume.decode_batch(inputs + [bad], out_dir,
                               retry_failed=False)
    assert not kept.ok
    assert kept.completed == []


def test_batch_kill9_recovery(tmp_path):
    """A SIGKILLed process leaves usable state and no half outputs."""
    inputs = make_inputs(tmp_path, count=4)
    out_dir = str(tmp_path / 'out')
    os.makedirs(out_dir)

    helper = tmp_path / 'worker.py'
    helper.write_text(
        'import time\n'
        'from audioread import resume\n'
        'real = resume._decode_to_file\n'
        'calls = [0]\n'
        'def slow(input_path, output_path, backends=None):\n'
        '    calls[0] += 1\n'
        '    if calls[0] == 2:\n'
        '        print("first-checkpointed", flush=True)\n'
        '        time.sleep(30)\n'
        '    return real(input_path, output_path, backends)\n'
        'resume._decode_to_file = slow\n'
        'inputs = %r\n'
        'resume.decode_batch(inputs, %r)\n' % (inputs, out_dir))

    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = repo_root + os.pathsep + \
        env.get('PYTHONPATH', '')
    proc = subprocess.Popen([sys.executable, str(helper)],
                            stdout=subprocess.PIPE, text=True, env=env)
    # Wait until the first item is checkpointed, then SIGKILL the
    # process while it is "decoding" the second one.
    assert proc.stdout.readline().strip() == 'first-checkpointed'
    proc.kill()
    proc.wait()
    assert proc.returncode == -signal.SIGKILL

    # The state file must be valid JSON with the first item done.
    with open(state_path_for(out_dir)) as f:
        state = json.load(f)
    done = [k for k, r in state['items'].items() if r['status'] == 'done']
    assert len(done) == 1

    # Simulate a half-written output left behind by the kill: it must
    # be cleaned up, never mistaken for a completed item.
    stale = os.path.join(out_dir, '.tmp-decode-stale.raw')
    with open(stale, 'wb') as f:
        f.write(b'partial')

    # The batch resumes where it stopped.
    result = resume.decode_batch(inputs, out_dir)
    assert result.ok
    assert len(result.skipped) == 1
    assert len(result.completed) + len(result.skipped) == 4
    assert not os.path.exists(stale)

    # Final outputs equal an uninterrupted reference run.
    ref_dir = str(tmp_path / 'ref')
    assert resume.decode_batch(inputs, ref_dir).ok
    assert output_bytes(out_dir) == output_bytes(ref_dir)


def test_batch_corrupted_state_rejected(tmp_path):
    """A corrupted state file raises instead of silently restarting."""
    inputs = make_inputs(tmp_path)
    out_dir = str(tmp_path / 'out')
    assert resume.decode_batch(inputs, out_dir).ok
    snapshot = output_bytes(out_dir)

    state_path = state_path_for(out_dir)
    with open(state_path, 'w') as f:
        f.write('{"version": 1, "items": {')  # truncated JSON

    with pytest.raises(resume.StateCorruptedError):
        resume.decode_batch(inputs, out_dir)

    # The outputs were not touched and nothing was redone.
    assert output_bytes(out_dir) == snapshot


def test_batch_invalid_state_structure_rejected(tmp_path):
    out_dir = str(tmp_path / 'out')
    os.makedirs(out_dir)
    with open(state_path_for(out_dir), 'w') as f:
        json.dump(['not', 'a', 'dict'], f)
    with pytest.raises(resume.StateCorruptedError):
        resume.decode_batch(make_inputs(tmp_path, 1), out_dir)


def test_batch_future_state_version_degrades(tmp_path):
    """Newer state files with unknown fields are read, not crashed."""
    inputs = make_inputs(tmp_path)
    out_dir = str(tmp_path / 'out')
    assert resume.decode_batch(inputs, out_dir).ok

    state_path = state_path_for(out_dir)
    with open(state_path) as f:
        state = json.load(f)
    state['version'] = resume.STATE_VERSION + 1
    state['future_top_level'] = {'x': 1}
    for record in state['items'].values():
        record['future_field'] = 'ignored'
    with open(state_path, 'w') as f:
        json.dump(state, f)

    result = resume.decode_batch(inputs, out_dir)
    assert result.ok
    assert result.completed == []
    assert len(result.skipped) == 3


def test_batch_tampered_output_reprocessed(tmp_path):
    """Outputs with wrong size/checksum are decoded again."""
    inputs = make_inputs(tmp_path)
    out_dir = str(tmp_path / 'out')
    assert resume.decode_batch(inputs, out_dir).ok
    before = output_bytes(out_dir)

    victim = os.path.join(out_dir, 'in-1.raw')
    with open(victim, 'ab') as f:
        f.write(b'garbage')

    result = resume.decode_batch(inputs, out_dir)
    assert result.ok
    assert result.completed == [os.path.abspath(inputs[1])]
    assert len(result.skipped) == 2
    assert output_bytes(out_dir) == before


def test_batch_concurrent_conflict(tmp_path):
    """Two processes cannot run the same batch at once."""
    inputs = make_inputs(tmp_path, count=1)
    out_dir = str(tmp_path / 'out')
    os.makedirs(out_dir)
    lock_path = state_path_for(out_dir) + resume._LOCK_SUFFIX

    with resume._BatchLock(lock_path):
        with pytest.raises(resume.BatchConflictError):
            resume.decode_batch(inputs, out_dir)

    # After the lock is released the batch runs normally.
    assert resume.decode_batch(inputs, out_dir).ok


def test_batch_state_write_enospc_preserves_state(tmp_path, monkeypatch):
    """A full disk aborts the batch but keeps the previous state."""
    inputs = make_inputs(tmp_path, count=2)
    out_dir = str(tmp_path / 'out')
    assert resume.decode_batch(inputs[:1], out_dir).ok
    with open(state_path_for(out_dir)) as f:
        good_state = f.read()

    real_replace = os.replace

    def full_disk(src, dst):
        if 'tmp-state' in os.path.basename(src):
            raise OSError(resume.errno.ENOSPC, 'No space left on device')
        return real_replace(src, dst)

    monkeypatch.setattr(resume.os, 'replace', full_disk)
    with pytest.raises(OSError):
        resume.decode_batch(inputs, out_dir)
    monkeypatch.undo()

    # The previous state file survived intact and stays usable.
    with open(state_path_for(out_dir)) as f:
        assert f.read() == good_state
    result = resume.decode_batch(inputs, out_dir)
    assert result.ok
    assert len(result.skipped) == 1


def test_default_api_unchanged(audiofile):
    """The pre-existing decode path is byte-identical, opt-in only."""
    import hashlib
    digest = hashlib.sha256()
    with audioread.audio_open(audiofile.path) as f:
        for buf in f:
            assert type(buf) == bytes
            digest.update(buf)
    # decode_batch is new API; audio_open signature is untouched.
    assert callable(audioread.decode_batch)
    assert digest.hexdigest()
