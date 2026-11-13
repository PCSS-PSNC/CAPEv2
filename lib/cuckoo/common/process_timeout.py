"""Run an arbitrary callable with a wall-clock timeout.

In-process/thread-based timeouts cannot reliably interrupt CPU-bound,
third-party native code (e.g. FLARE-CAPA/vivisect): an abandoned thread keeps
burning CPU and ``signal.alarm`` only fires between bytecodes. The only
dependable way to respect a budget is to run the work in a separate process
and kill it out-of-band when the deadline passes.

This module follows the fork + kill pattern already used for the extractor
pool in ``lib.cuckoo.common.integrations.file_extra_info``. The child ships
its return value (or re-raised exception) back over a pipe so the caller sees
the same behaviour as a plain synchronous call -- except that an overdue child
is terminated and ``TimeoutError`` is raised instead.

On platforms without ``os.fork`` (e.g. Windows) the deadline cannot be
enforced, so ``run_with_timeout`` degrades to a plain synchronous call.
"""

import logging
import os
import pickle
import select
import signal
import time
from contextlib import suppress

log = logging.getLogger(__name__)


class ResultUnavailableError(Exception):
    """The forked child died or produced a malformed frame before delivering
    its payload (e.g. OOM-killed or segfaulted mid-write). This is distinct
    from a deadline expiry, which is reported as ``None`` by ``_read_payload``
    and surfaced as ``TimeoutError`` by ``run_with_timeout``.
    """


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _child_main(fn, args, kwargs, write_fd):
    """Child entry point: run ``fn``, ship the outcome, then hard-exit."""
    try:
        # Start a new session/process group so the parent's killpg() only ever
        # targets this child (and anything it spawns), never the parent itself.
        with suppress(OSError):
            os.setsid()
        try:
            result = fn(*args, **kwargs)
            payload = pickle.dumps(("ok", result), protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as exc:  # noqa: BLE001 - nothing may escape the child
            try:
                payload = pickle.dumps(("raise", exc), protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                wrapped = RuntimeError(f"{type(exc).__name__}: {exc}")
                payload = pickle.dumps(("raise", wrapped), protocol=pickle.HIGHEST_PROTOCOL)
        _write_all(write_fd, len(payload).to_bytes(8, "big") + payload)
    finally:
        # Never return into the forked parent's heap.
        os._exit(0)


def _read_payload(read_fd, deadline):
    """Read the child's framed payload.

    The wire format is an 8-byte big-endian length header followed by that many
    bytes of payload (see ``_child_main``). Returns the payload bytes without
    the header.

    Returns ``None`` if the deadline elapsed before the child finished
    delivering the frame. Raises :class:`ResultUnavailableError` if the stream reaches
    EOF without a complete, well-formed frame -- i.e. the child crashed or was
    killed mid-write and no usable result ever arrived.

    Reads accumulate until EOF (the child always closes its write end and
    exits -- the single-writer invariant guarantees EOF fires), then parse the
    leading header and validate the promised length against what arrived. Read
    boundaries never align with the framing, so all bytes are gathered first
    and the header is only used to validate, not to size individual reads.
    """
    buf = b""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        readable, _, _ = select.select([read_fd], [], [], remaining)
        if not readable:
            return None
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break  # EOF: child finished (or died) and closed the pipe
        buf += chunk
    if len(buf) < 8:
        raise ResultUnavailableError("child died before sending its payload header")
    size = int.from_bytes(buf[:8], "big")
    if len(buf) - 8 < size:
        raise ResultUnavailableError(f"child died mid-transfer: got {len(buf) - 8} of {size} promised bytes")
    if size == 0:
        raise ResultUnavailableError("malformed frame: payload length is zero")
    return buf[8 : 8 + size]


def _kill_child(pid):
    """Kill the child (and any grandchildren it spawned).

    Prefer a process-group kill so potential grandchildren spawned by the child are swept too.
    If the child never detached into its own process group (its ``setsid`` was skipped/failed) and still shares ours,
    fall back to killing just the pid so we never risk terminating our own process group.
    """
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    if pgid is not None and pgid != os.getpgrp():
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    with suppress(ProcessLookupError, OSError):
        os.kill(pid, signal.SIGKILL)


def run_with_timeout(fn, timeout, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` enforcing a wall-clock ``timeout`` (seconds).

    - ``timeout`` falsy (``None`` / ``0`` / negative): run synchronously in the
      current process with no extra budget -- behaviour is unchanged from a
      plain call.
    - Platforms without ``os.fork`` (e.g. Windows): run synchronously, as the
      deadline cannot be enforced there.
    - Otherwise fork ``fn`` into a child. The child's return value (or raised
      exception) is returned / re-raised in the parent, identically to a
      synchronous call. If the child exceeds ``timeout`` seconds it is killed
      and ``TimeoutError`` is raised. If ``os.fork`` itself fails, the call
      degrades to a synchronous run. If the child dies (crash/OOM) without
      delivering a complete frame, :class:`ResultUnavailableError` is raised.

    The child (and any process group it spawned) is terminated and reaped on
    every parent-side exit path -- success, timeout, ``ResultUnavailableError``, or any
    unexpected exception -- so nothing it forked outlives the call.

    Note: forking inside a process that is not single-threaded skips a full
    Python interpreter re-initialisation in the child. The work ``fn`` must
    therefore avoid re-entering locks that another thread may hold at fork
    time. This is the same constraint the existing extractor pool relies on.
    """
    if not timeout or timeout <= 0:
        return fn(*args, **kwargs)

    if not hasattr(os, "fork"):
        log.warning("os.fork() unavailable (non-POSIX platform); cannot enforce timeout, running synchronously")
        return fn(*args, **kwargs)

    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
    except OSError:
        # fork() failed: no child was created, so there is nothing to reap.
        # Release the pipe and degrade to a synchronous run rather than
        # failing outright.
        log.warning("os.fork() failed; running synchronously", exc_info=True)
        with suppress(OSError):
            os.close(read_fd)
        with suppress(OSError):
            os.close(write_fd)
        return fn(*args, **kwargs)

    if pid == 0:
        # Child: run the work, ship the outcome, then hard-exit. Never returns.
        os.close(read_fd)
        _child_main(fn, args, kwargs, write_fd)
        os._exit(127)  # defensive; _child_main should always os._exit()s

    # Parent. Everything from here is wrapped in try/finally so that any
    # exception raised between the fork and the read still releases the pipe
    # fds and reaps the child.
    try:
        deadline = time.monotonic() + float(timeout)
        os.close(write_fd)  # Leave the child as the single writer, so EOF on the read end signals.
        payload = _read_payload(read_fd, deadline)
        if payload is None:
            raise TimeoutError(f"process timed out after {timeout}s")
        status, value = pickle.loads(payload)
        if status == "raise":
            raise value
        return value
    finally:
        # Sweep the child's process group on every exit path (success, timeout, ResultUnavailableError, or other exception).
        _kill_child(pid)
        with suppress(OSError):
            os.close(read_fd)
        with suppress(OSError):
            os.close(write_fd)
        with suppress(ChildProcessError, OSError):
            os.waitpid(pid, 0)  #  SIGKILL used by _kill_child may not terminate child
            # so a still- alive child (timeout path) can be reaped rather than blocking forever.
