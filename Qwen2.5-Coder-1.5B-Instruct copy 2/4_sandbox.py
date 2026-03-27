import sys
import signal
import resource
import traceback


def timeout_handler(signum, frame):
    raise TimeoutError("Execution exceeded the 5-second hard limit.")


def set_limits():
    # Set Memory Limit to 1GB (1024 * 1024 * 1024 bytes)
    gigabyte = 1024 * 1024 * 1024
    # resource.setrlimit(resource.RLIMIT_AS, (gigabyte, gigabyte))

    # Leave the CPU Timeout active:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)


if __name__ == "__main__":
    # Read the AI-generated code passed from the main controller
    code_to_run = sys.stdin.read()

    try:
        set_limits()
        exec(code_to_run, {})
        print("SUCCESS")

    except TimeoutError:
        sys.stderr.write("TIMEOUT_ERROR\n")
        sys.exit(1)

    except AssertionError:
        # --- THE MAGIC FIX ---
        # Look at the stack trace to see exactly WHERE the AssertionError happened
        _, _, tb = sys.exc_info()
        frames = traceback.extract_tb(tb)

        # HumanEval tests are always wrapped in a function named 'check'
        # If the error happened inside 'check', the AI returned the wrong answer!
        if any(f.name == 'check' for f in frames):
            sys.stderr.write("WRONG_OUTPUT_ERROR\n")
        else:
            # If it happened elsewhere, the AI wrote a failing assert in its own code
            sys.stderr.write("ASSERTION_ERROR\n")
        sys.exit(2)

    except Exception:
        sys.stderr.write("EXCEPTION_ERROR\n")
        sys.exit(3)