from llm_sdk import Small_LLM_Model

from .constrained_decoder import ConstrainedDecoder
from .file_io import load_functions_definitions
from .function_caller import FunctionCaller
from .vocabulary import Vocabulary
from .models import TestPrompt


print("Loading model...")
model = Small_LLM_Model()

vocabulary = Vocabulary.from_file(
    model.get_path_to_vocab_file()
)

decoder = ConstrainedDecoder(
    model,
    vocabulary
)

functions = load_functions_definitions(
    "data/input/functions_definition.json"
)

caller = FunctionCaller(
    decoder,
    functions
)

print("Ready")

while True:
    text = input("\nPrompt: ")

    if text == "exit":
        break

    result = caller.call(
        TestPrompt(prompt=text)
    )

    print(result.model_dump_json(indent=2))