import subprocess
from collections import Counter


STATUS_MARKER = "__SANDBOX_STATUS__:"


def run_in_sandbox(generated_code: str, test_code: str) -> str:
    """Passes code to the sandbox and classifies the outcome."""
    full_code = generated_code + "\n\n" + test_code

    try:
        # Spawn the sandbox process securely
        result = subprocess.run(
            ["python3", "4_sandbox.py"],  # <--- This requires sandbox.py to be in the folder!
            input=full_code,
            text=True,
            capture_output=True,
            timeout=7
        )

        combined_output = f"{result.stdout}\n{result.stderr}"

        # Parse the sandbox status marker first so traceback text does not confuse classification.
        if result.returncode == 0:
            return "SUCCESS"
        elif f"{STATUS_MARKER}TIMEOUT_ERROR" in combined_output:
            return "TIMEOUT"
        elif f"{STATUS_MARKER}WRONG_OUTPUT_ERROR" in combined_output:
            return "WRONG_OUTPUT"
        elif f"{STATUS_MARKER}ASSERTION_ERROR" in combined_output:
            return "ASSERTION_ERROR"
        elif f"{STATUS_MARKER}EXCEPTION_ERROR" in combined_output:
            print("\n--- DEBUG: HIDDEN EXCEPTION CAUGHT ---")
            print(combined_output.strip())
            print("---------------------------------------\n")
            return "EXCEPTION"
        else:
            if "No such file" in combined_output or "can't open file" in combined_output:
                print(f"\n[SYSTEM ERROR] {combined_output.strip()}")
                return "SYSTEM_ERROR"
            print("\n--- DEBUG: UNCLASSIFIED SANDBOX OUTPUT ---")
            print(combined_output.strip())
            print("-----------------------------------------\n")
            return "SYSTEM_ERROR"

    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def measure_1000_generations(predictions, references):
    """
    Simulates testing 1000 generated functions and calculates the % breakdown.
    """
    print(f"Executing {len(predictions)} sandboxed tests...\n")

    results = []
    for gen_code, test in zip(predictions, references):
        outcome = run_in_sandbox(gen_code, test)
        results.append(outcome)

    # Tally the results
    counts = Counter(results)
    total = len(results)

    print("=== SYSTEMATIC FAILURE ANALYSIS ===")
    print(f"Total Evaluated: {total}")
    print(f"Success Rate:    {(counts.get('SUCCESS', 0) / total) * 100:.2f}%")
    print("-" * 35)
    print(f"Timeouts:        {(counts.get('TIMEOUT', 0) / total) * 100:.2f}%")
    print(f"Exceptions:      {(counts.get('EXCEPTION', 0) / total) * 100:.2f}%")
    print(f"Assertion Errs:  {(counts.get('ASSERTION_ERROR', 0) / total) * 100:.2f}%")
    print(f"Wrong Outputs:   {(counts.get('WRONG_OUTPUT', 0) / total) * 100:.2f}%")

    if "SYSTEM_ERROR" in counts:
        print(f"\n⚠️ WARNING: {counts['SYSTEM_ERROR']} tests failed because sandbox.py was not found!")


# --- Example Usage ---
if __name__ == "__main__":
    human_eval_test = """
def check(candidate):
    assert candidate([]) == False
    assert candidate([1, 2, -3, 1, 2, -3]) == False
    assert candidate([1, 2, -4, 5, 6]) == True
    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False
    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True
    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True

check(below_zero)
"""

    mock_predictions = [
        # 1. SUCCESS
        """
def below_zero(operations: list) -> bool:
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
        """,
        # 2. TIMEOUT
        """
def below_zero(operations: list) -> bool:
    balance = 0
    i = 0
    while i < len(operations):
        balance += operations[i]
    return False
        """,
        # 3. EXCEPTION
        """
def below_zero(operations: list) -> bool:
    balance = "Zero" 
    for op in operations:
        balance += op 
    return False
        """,
        # 4. ASSERTION ERROR
        """
def below_zero(operations: list) -> bool:
    if sum(operations) < 0:
        return True
    return False
        """,
        # 5. EXCEPTION (Memory Leak caught by 1GB limit)
        """
def below_zero(operations: list) -> bool:
    giant_list = [0] * (10**9) 
    return False
        """
    ]

    mock_tests = [human_eval_test] * 5
    measure_1000_generations(mock_predictions, mock_tests)
