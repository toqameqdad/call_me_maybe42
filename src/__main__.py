"""Command-line entry point for the Call Me Maybe project."""

import argparse
import sys

from llm_sdk import Small_LLM_Model

from .file_io import (
    load_functions_definitions,
    load_test_prompts,
    save_results,
)
from .function_caller import FunctionCaller
from .llm_decoder import LLMDecoder


DEFAULT_FUNCTIONS_PATH = "data/input/functions_definition.json"
DEFAULT_INPUT_PATH = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = (
    "data/output/function_calling_results.json"
)


def _parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="call-me-maybe",
        description=(
            "Translate natural-language prompts into "
            "structured function calls."
        ),
    )

    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_PATH,
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to the test prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the results JSON file.",
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    """Run the function-calling pipeline."""

    args = _parse_args(argv)

    try:
        functions = load_functions_definitions(
            args.functions_definition
        )

        prompts = load_test_prompts(
            args.input
        )

    except RuntimeError as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 1


    try:
        print(
            "Loading model, this may take a moment..."
        )

        model = Small_LLM_Model()

        decoder = LLMDecoder(
            model
        )

        caller = FunctionCaller(
            decoder,
            functions,
        )

    except (
        OSError,
        ValueError,
        RuntimeError,
    ) as error:

        print(
            f"Error: Unable to initialize the model: {error}",
            file=sys.stderr,
        )

        return 1


    results = []

    for index, test_prompt in enumerate(
        prompts,
        start=1,
    ):

        print(
            f"[{index}/{len(prompts)}] "
            f"{test_prompt.prompt}"
        )

        try:

            results.append(
                caller.call(test_prompt)
            )

        except ValueError as error:

            print(
                f"  Skipped: {error}",
                file=sys.stderr,
            )


    try:

        save_results(
            args.output,
            results,
        )

    except RuntimeError as error:

        print(
            f"Error: {error}",
            file=sys.stderr,
        )

        return 1


    print(
        f"Wrote {len(results)} result(s) "
        f"to {args.output}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
