from src.file_io import (
    load_functions_definitions,
    load_test_prompts,
    save_results,
)
from src.models import FunctionCallResult


functions = load_functions_definitions("data/test/functions.json")
prompts = load_test_prompts("data/test/prompts.json")

print(functions)
print(prompts)

results = [
    FunctionCallResult(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": 2, "b": 3},
    )
]

save_results("data/test/results.json", results)

print("\nResults saved!")