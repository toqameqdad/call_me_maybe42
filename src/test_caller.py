from llm_sdk import Small_LLM_Model

from .file_io import load_functions_definitions
from .function_caller import FunctionCaller
from .llm_decoder import LLMDecoder
from .models import TestPrompt


print("Loading model...")

model = Small_LLM_Model()

decoder = LLMDecoder(model)

functions = load_functions_definitions(
    "data/input/functions_definition.json"
)

caller = FunctionCaller(
    decoder,
    functions,
)

print("Ready")

result = caller.call(
    TestPrompt(
        prompt="Replace all vowels in 'Programming is fun' with asterisks"
    )
)

print(result.model_dump_json(indent=2))