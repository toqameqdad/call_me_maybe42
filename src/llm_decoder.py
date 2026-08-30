"""Simple LLM decoder wrapper."""

import json

from llm_sdk import Small_LLM_Model


class LLMDecoder:
    """Wrapper around the local language model."""

    def __init__(
        self,
        model: Small_LLM_Model,
    ) -> None:
        self.model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        """Generate text continuation from the model."""

        input_ids = self.model.encode(prompt)

        generated_ids = []

        for _ in range(max_tokens):

            current_ids = (
                input_ids[0].tolist()
                + generated_ids
            )

            logits = self.model.get_logits_from_input_ids(
                current_ids
            )

            next_token = max(
                range(len(logits)),
                key=lambda i: logits[i],
            )

            generated_ids.append(next_token)

            text = self.model.decode(
                generated_ids
            )

            # stop on JSON completion
            if self._looks_complete(text) and text.count("{") == text.count("}"):
                break

        return self.model.decode(
            generated_ids
        )

    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 512,
    ) -> dict:
        """Generate and parse JSON output."""

        text = self.generate(
            prompt,
            max_tokens,
        )

        text = self._repair_json(text)

        text = self._extract_json(text)

        return json.loads(text)

    def _extract_json(
        self,
        text: str,
    ) -> str:
        """Extract JSON object from generated text."""

        text = text.strip()

        # Remove markdown code blocks
        if "```" in text:
            parts = text.split("```")

            for part in parts:
                part = part.strip()

                if part.startswith("json"):
                    part = part[4:].strip()

                if part.startswith("{") and part.endswith("}"):
                    return part

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "No JSON object found."
            )

        return text[start:end + 1]

    def _repair_json(
        self,
        text: str,
    ) -> str:
        """Repair incomplete JSON output."""

        text = text.strip()

        # Remove markdown if exists
        if "```" in text:
            parts = text.split("```")

            for part in parts:
                part = part.strip()

                if part.startswith("json"):
                    part = part[4:].strip()

                if part.startswith("{"):
                    text = part
                    break

        # Close unfinished string
        quote_count = text.count('"')

        if quote_count % 2 != 0:
            text += '"'

        # Remove incomplete last key/value
        if text.rstrip().endswith(":"):
            text = text.rstrip()[:-1]

        # Close JSON object
        while text.count("{") > text.count("}"):
            text += "}"

        return text

    def _looks_complete(
        self,
        text: str,
    ) -> bool:
        """Check if generation likely finished."""

        text = text.strip()

        if text.endswith("}"):
            return True

        if text.endswith("]"):
            return True

        return False
