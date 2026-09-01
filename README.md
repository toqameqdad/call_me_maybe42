*This activity has been created as part of the 42 curriculum by tmeqdad.*

# Call Me Maybe

## Description

**Call Me Maybe** is a function calling system that translates natural language requests into structured function calls.

Large Language Models (LLMs) are good at understanding and generating natural language, but they do not naturally produce reliable, machine-executable structured output. Function calling bridges this gap by converting a user's request into a function name and correctly typed arguments.

For example, given:

```text
What is the sum of 40 and 2?
```

the system should produce a function call rather than directly answering `42`:

```json
{
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 40,
        "b": 2
    }
}
```

The project uses the **Qwen/Qwen3-0.6B** model together with the provided `llm_sdk` and constrained decoding.

The system:

1. Loads the available function definitions.
2. Reads natural-language prompts.
3. Uses the LLM to determine which function should be called.
4. Extracts the required arguments.
5. Uses constrained decoding to restrict token generation.
6. Produces JSON that follows the required function schema.
7. Validates the resulting data structures using Pydantic.

The input function definitions are dynamic, so the implementation must not be hardcoded to the example functions provided with the activity.

---

# Features

* Natural-language to function-call conversion.
* LLM-based function selection.
* Automatic argument extraction.
* Dynamic function definitions.
* Schema-aware constrained decoding.
* Pydantic validation.
* JSON input and output handling.
* Typed function arguments.
* Graceful error handling.
* Configurable input and output paths.

---

# Instructions

## Requirements

The project requires:

* Python 3.10 or later.
* `uv` package manager.
* `numpy`.
* `pydantic`.
* The provided `llm_sdk` package.

The project must work with:

```text
Qwen/Qwen3-0.6B
```

The project uses the provided `llm_sdk` rather than directly depending on external LLM frameworks.

The following libraries and approaches are not used:

* `dspy`
* `pytorch`
* `huggingface`
* `transformers`
* `outlines`
* Similar packages that replace the required implementation.

The function selection is performed by the LLM and is not implemented using keyword matching, hardcoded rules, or other heuristics.

---

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd call_me_maybe
```

Install the project dependencies:

```bash
uv sync
```

The project uses `uv` to create and manage the Python environment and install the required dependencies.

The provided `llm_sdk` directory should be available alongside the `src` directory.

---

## Running the Program

Run the program with the default paths:

```bash
uv run python -m src
```

By default, the program reads:

```text
data/input/
```

and writes the generated results to:

```text
data/output/
```

Custom paths can be supplied with:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

---

# Project Structure

```text
call_me_maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── models.py
│   ├── io_utils.py
│   ├── vocabulary.py
│   ├── pipeline.py
│   ├── constraine.py
│   └── decoder.py
│
├── llm_sdk/
│   └── ...
│
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
│
├── Makefile
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

The `data/output/` directory is generated when the program runs and should not be included in the repository.

---

# Algorithm Explanation

## Function Calling Pipeline

The system converts a natural-language prompt into a structured function call.

The general pipeline is:

```text
Natural Language Prompt
        ↓
      LLM
        ↓
Function Selection
        ↓
Argument Extraction
        ↓
Constrained Decoding
        ↓
Schema-Compliant JSON
        ↓
     Validation
```

The project does not simply ask the model to generate JSON and trust its response. Small language models can produce malformed or incorrect structured output, so generation is constrained token-by-token.

---

## LLM Generation Pipeline

The LLM generation process follows these steps:

```text
Prompt
  ↓
Tokenization
  ↓
Input IDs
  ↓
LLM
  ↓
Logits
  ↓
Next Token Selection
```

### 1. Prompt

The original natural-language request is provided to the model.

Example:

```text
What is the sum of 2 and 3?
```

### 2. Tokenization

The tokenizer converts the text into tokens.

Tokens are subword units rather than necessarily complete words. They can also preserve spaces and punctuation.

### 3. Input IDs

Each token is mapped to a numerical token ID that can be processed by the model.

### 4. LLM Processing

The model processes the input token IDs and produces logits representing scores for possible next tokens.

### 5. Logits

The model produces a score for every possible token in its vocabulary.

For example:

```text
token A → 0.02
token B → 0.85
token C → 0.01
...
```

### 6. Next Token Selection

A valid next token is selected based on the model's output.

After selecting a token, it is added to the generated sequence and the process repeats.

---

# Constrained Decoding

## Why Constrained Decoding Is Needed

The Qwen3-0.6B model is relatively small and cannot be expected to reliably generate correctly structured JSON every time.

The project therefore uses constrained decoding to control the generation process token-by-token.

The goal is not simply to validate the generated JSON after generation.

Instead, invalid tokens are prevented from being selected during generation.

---

## Constrained Decoding Process

At every generation step:

1. The LLM produces logits for all vocabulary tokens.
2. The current generated text is checked against the expected JSON structure.
3. The decoder determines which tokens can legally continue the current state.
4. Tokens that would violate the JSON structure or function schema are rejected.
5. The logits of invalid tokens are restricted, effectively preventing their selection.
6. A valid token is selected.
7. The state is updated.
8. The process continues until the complete function call is generated.

Conceptually:

```text
                 LLM
                  ↓
               Logits
                  ↓
        ┌──────────────────┐
        │ Constraint Check │
        └──────────────────┘
             ↓         ↓
          Valid       Invalid
          Tokens       Tokens
             ↓           ↓
       Can be chosen   Rejected
             ↓
       Next Token
```

The vocabulary file supplied through the LLM SDK is important because it maps token IDs to their corresponding token representations. This allows the decoder to determine which tokens are valid at each state.

---

## Schema-Aware Constraints

The decoder must enforce more than valid JSON syntax.

It must also respect the function definition.

For example, if a function requires:

```json
{
    "a": {
        "type": "number"
    }
}
```

the generated value must be a valid number rather than an arbitrary string.

Similarly, required fields must be present and their values must have the types specified by the function definition.

The state machine therefore keeps track of both:

* JSON structure.
* Expected function schema.

This allows constrained decoding to maintain schema compliance during generation.

---

# Dynamic Function Definitions

Available functions are loaded from:

```text
data/input/functions_definition.json
```

Each function definition contains information such as:

* Function name.
* Description.
* Parameters.
* Parameter types.
* Return type.

Example:

```json
{
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
        "a": {
            "type": "number"
        },
        "b": {
            "type": "number"
        }
    },
    "returns": {
        "type": "number"
    }
}
```

The implementation does not hardcode the available functions.

This is important because the input function definitions can change during peer evaluation.

---

# Pydantic Validation

Pydantic is used for validation of the project's data structures.

The project uses typed models to represent structured data such as:

* Function definitions.
* Function parameters.
* Test prompts.
* Generated function calls.

Validation helps detect invalid input data and prevents malformed structures from silently propagating through the program.

All project classes use Pydantic validation as required by the activity.

---

# Input Files

The project processes two main input files.

## `function_calling_tests.json`

Contains the natural-language prompts that the system must process.

Example:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?"
    },
    {
        "prompt": "Greet john"
    },
    {
        "prompt": "Reverse the string hello"
    }
]
```

## `functions_definition.json`

Contains the functions that are available to the LLM.

The system must use the definitions supplied in this file rather than relying on hardcoded functions.

Input files may contain invalid JSON or may be missing, so the program handles these situations with clear error messages.

---

# Output Format

The generated result is written to:

```text
data/output/function_calling_results.json
```

The output is a JSON array.

Each result contains exactly:

```text
prompt
name
parameters
```

Example:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2,
            "b": 3
        }
    }
]
```

The output must:

* Be valid JSON.
* Contain no comments.
* Contain no additional prose.
* Contain no unexpected keys.
* Include all required arguments.
* Use the types specified by the function definition.

---

# Design Decisions

## LLM-Based Function Selection

Function selection is performed by the LLM.

The implementation does not use:

* Keyword matching.
* Hardcoded function mappings.
* Manual rules based on prompt text.

This allows the system to work with different function definitions and prompts.

---

## Dynamic Schemas

Function definitions are loaded dynamically from JSON.

This avoids coupling the implementation to the example functions and allows the reviewer to provide different function sets.

---

## State Machine

A state machine is used to keep track of the current position in the expected JSON structure.

The state determines which tokens can legally continue the generated output.

This provides the foundation for schema-aware constrained decoding.

---

## Vocabulary-Based Token Validation

The vocabulary supplied by the LLM SDK is used to map token IDs to token strings.

The decoder uses this information when determining which vocabulary tokens can continue the current constrained state.

---

## Error Handling

The application handles errors such as:

* Missing input files.
* Invalid JSON.
* Invalid function definitions.
* Invalid test prompts.
* Generation failures.
* Invalid generated structures.

Errors should be reported clearly rather than causing an unexpected program crash.

---

# Performance Analysis

The project targets the performance requirements specified by the activity.

## Accuracy

The target is:

* 90%+ correct function selection.
* 90%+ correct argument extraction.

## Reliability

Constrained decoding is designed to ensure that generated output:

* Is valid JSON.
* Follows the expected schema.
* Contains correctly typed arguments.

The target is **100% valid, schema-compliant JSON output**.

## Speed

The complete set of test prompts should be processed in under:

```text
5 minutes
```

on standard hardware.

## Flexibility

Because functions and their schemas are loaded dynamically, the implementation can support different function sets without modifying the source code.

---

# Challenges Faced

## Structured JSON Generation

Small language models are not always reliable at generating structured JSON.

The solution was to move structural control from prompting into the decoding process.

Instead of allowing every vocabulary token to be selected, the constrained decoder only allows tokens that can maintain the expected JSON structure and schema.

---

## Schema Compliance

Valid JSON alone is not sufficient.

For example, this is valid JSON:

```json
{
    "a": "hello"
}
```

but it is incorrect if `a` is required to be a number.

The decoder therefore considers the expected parameter type when determining valid continuations.

---

## Dynamic Function Sets

The provided examples are not guaranteed to be the same during peer evaluation.

The implementation therefore loads function definitions dynamically and avoids hardcoded function names and arguments.

---

## Invalid Input

Input files can be missing or contain malformed JSON.

The file handling layer validates input and reports errors clearly instead of allowing the application to fail unexpectedly.

---

# Testing Strategy

The implementation should be tested using different categories of prompts.

## Mathematical Operations

Example:

```text
What is the sum of 2 and 3?
```

Expected behavior:

```text
fn_add_numbers
```

with the appropriate numeric arguments.

## String Operations

Example:

```text
Reverse the string hello
```

The system should select the appropriate string-processing function and extract:

```json
{
    "s": "hello"
}
```

## Multiple Functions

Tests should include multiple available functions with different descriptions and parameters to verify that the LLM selects the correct function.

## Edge Cases

Testing should cover:

* Empty strings.
* Large numbers.
* Special characters.
* Different parameter types.
* Missing arguments.
* Invalid JSON.
* Ambiguous prompts.
* Functions with multiple parameters.
* Different function definitions.

The input files should be changed during testing to ensure the implementation is not dependent on the provided examples.

---

# Example Usage

Place the input files in:

```text
data/input/
```

For example:

```text
data/input/
├── functions_definition.json
└── function_calling_tests.json
```

Then run:

```bash
uv run python -m src
```

The generated file will be:

```text
data/output/function_calling_results.json
```

For custom files:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

---

# Makefile

The project provides a `Makefile` for common development tasks.

Run the program:

```bash
make run
```

Install dependencies:

```bash
make install
```

Run the program using Python's debugger:

```bash
make debug
```

Remove generated caches and temporary files:

```bash
make clean
```

Run the required static checks:

```bash
make lint
```

The lint command includes:

```text
flake8 .
mypy . --warn-return-any --warn-unused-ignores \
    --ignore-missing-imports \
    --disallow-untyped-defs \
    --check-untyped-defs
```

A stricter check can also be provided:

```bash
make lint-strict
```

---

# Resources

## Python Documentation

Python documentation:

https://docs.python.org/3/

## Pydantic Documentation

Pydantic documentation:

https://docs.pydantic.dev/

## JSON

JSON specification and documentation:

https://www.json.org/

## Qwen

Qwen model information:

https://huggingface.co/Qwen

## AI Usage

AI tools were used as a learning and development aid during this project.

They were used for:

* Understanding LLM function calling concepts.
* Understanding the LLM generation pipeline.
* Understanding tokenization, input IDs, logits, and next-token selection.
* Exploring constrained decoding concepts.
* Reviewing implementation ideas.
* Debugging errors.
* Improving documentation.
* Identifying potential edge cases.

AI-generated suggestions were reviewed, tested, and adapted manually. The implementation was not accepted blindly, and the developer remains responsible for understanding and explaining the final code.

---

# Conclusion

Call Me Maybe demonstrates how a small language model can be used to translate natural-language requests into structured function calls.

The main challenge is not simply asking the LLM to produce JSON. Instead, the project uses constrained decoding to control generation token-by-token while respecting both JSON structure and the function schema.

The combination of:

```text
LLM
+
Vocabulary
+
State Machine
+
Constrained Decoding
+
Pydantic Validation
```

provides a structured pipeline for producing reliable function calls from natural-language prompts.
