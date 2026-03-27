import subprocess

STATUS_MARKER = "__SANDBOX_STATUS__:"


def test_sandbox_case(name: str, code: str):
    print(f"\n--- Testing: {name} ---")

    try:
        result = subprocess.run(
            ["python3", "4_sandbox.py"],
            input=code,
            text=True,
            capture_output=True,
            timeout=7
        )

        # Print exactly what the sandbox spat out
        print(f"Return Code: {result.returncode}")
        print(f"STDOUT: {result.stdout.strip()}")
        print(f"STDERR: {result.stderr.strip()}")

        # Our classification logic
        combined_output = f"{result.stdout}\n{result.stderr}"

        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print("✅ CLASSIFICATION: SUCCESS")
        elif f"{STATUS_MARKER}TIMEOUT_ERROR" in combined_output:
            print("⏳ CLASSIFICATION: TIMEOUT")
        elif f"{STATUS_MARKER}WRONG_OUTPUT_ERROR" in combined_output:
            print("❌ CLASSIFICATION: WRONG_OUTPUT")
        elif f"{STATUS_MARKER}ASSERTION_ERROR" in combined_output:
            print("❌ CLASSIFICATION: ASSERTION_ERROR")
        elif f"{STATUS_MARKER}EXCEPTION_ERROR" in combined_output:
            print("💥 CLASSIFICATION: EXCEPTION")
        else:
            print("❓ CLASSIFICATION: WRONG_OUTPUT / UNKNOWN")

    except subprocess.TimeoutExpired:
        print("⏳ CLASSIFICATION: TIMEOUT (Caught by Subprocess)")


if __name__ == "__main__":
    # 1. Test a perfect function
    success_code = """
def add(a, b): return a + b
assert add(2, 2) == 4
"""
    test_sandbox_case("1. Clean Code (Should be SUCCESS)", success_code)

    # 2. Test an infinite loop
    timeout_code = """
while True:
    pass
"""
    test_sandbox_case("2. Infinite Loop (Should be TIMEOUT in ~5s)", timeout_code)

    # 3. Test a crash
    exception_code = """
x = 1 / 0
"""
    test_sandbox_case("3. Divide by Zero (Should be EXCEPTION)", exception_code)

    # 4. Test a failed unit test
    assertion_code = """
def add(a, b): return a + b
assert add(2, 2) == 5
"""
    test_sandbox_case("4. Bad Math Logic (Should be ASSERTION_ERROR)", assertion_code)
