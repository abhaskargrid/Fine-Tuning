import io
import os
import resource
import signal
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout


STATUS_MARKER = "__SANDBOX_STATUS__:"
MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024
TIMEOUT_SECONDS = 5


def timeout_handler(signum, frame):
    raise TimeoutError(f"Execution exceeded the {TIMEOUT_SECONDS}-second hard limit.")


def apply_resource_limits():
    """Apply hard limits inside the already-forked child process."""
    for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
        limit = getattr(resource, limit_name, None)
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        except (OSError, ValueError):
            # Some platforms reject one of these limits; keep the sandbox running.
            pass


def classify_assertion() -> str:
    _, _, tb = sys.exc_info()
    frames = traceback.extract_tb(tb)
    if any(frame.name == "check" for frame in frames):
        return "WRONG_OUTPUT_ERROR"
    return "ASSERTION_ERROR"


def execute_user_code(code_to_run: str):
    """Run untrusted code and emit a structured sandbox status."""
    apply_resource_limits()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    status = "SUCCESS"

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code_to_run, {})
    except TimeoutError:
        status = "TIMEOUT_ERROR"
        traceback.print_exc(file=stderr_buffer)
    except AssertionError:
        status = classify_assertion()
        traceback.print_exc(file=stderr_buffer)
    except BaseException:
        status = "EXCEPTION_ERROR"
        traceback.print_exc(file=stderr_buffer)
    finally:
        signal.alarm(0)

    sys.stdout.write(stdout_buffer.getvalue())
    sys.stderr.write(stderr_buffer.getvalue())
    if status == "SUCCESS":
        sys.stdout.write("SUCCESS\n")
    else:
        sys.stderr.write(f"{STATUS_MARKER}{status}\n")


def parent_wait(child_pid: int, read_fd: int):
    """Read child output and enforce a final watchdog in the parent."""
    chunks = []
    deadline = time.monotonic() + TIMEOUT_SECONDS + 2

    while True:
        try:
            pid, status = os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            pid, status = child_pid, 0

        try:
            chunk = os.read(read_fd, 4096)
            if chunk:
                chunks.append(chunk)
        except BlockingIOError:
            pass

        if pid == child_pid:
            break

        if time.monotonic() > deadline:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(child_pid, 0)
            chunks.append(f"{STATUS_MARKER}TIMEOUT_ERROR\n".encode())
            break

        time.sleep(0.01)

    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)

    os.close(read_fd)
    combined = b"".join(chunks).decode("utf-8", errors="replace")
    sys.stdout.write(combined)

    if f"{STATUS_MARKER}TIMEOUT_ERROR" in combined:
        sys.exit(1)
    if f"{STATUS_MARKER}WRONG_OUTPUT_ERROR" in combined or f"{STATUS_MARKER}ASSERTION_ERROR" in combined:
        sys.exit(2)
    if f"{STATUS_MARKER}EXCEPTION_ERROR" in combined:
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    code_to_run = sys.stdin.read()

    read_fd, write_fd = os.pipe()
    child_pid = os.fork()

    if child_pid == 0:
        os.close(read_fd)
        os.dup2(write_fd, sys.stdout.fileno())
        os.dup2(write_fd, sys.stderr.fileno())
        os.close(write_fd)
        try:
            execute_user_code(code_to_run)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    os.close(write_fd)
    parent_wait(child_pid, read_fd)
