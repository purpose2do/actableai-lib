from functools import lru_cache
from typing import Iterable

from actableai.parameters.numeric import FloatParameter
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters
from actableai.parameters.string import StringParameter
from actableai.text_extraction.models.base import BaseTextExtractionModel, Model
from actableai.utils.typing import JSONType


class OpenAI(BaseTextExtractionModel):
    @classmethod
    @lru_cache(maxsize=0)
    def get_parameters(cls) -> Parameters:
        parameters = [
            *cls.get_base_parameters(),
            OptionsParameter[str](
                name="model",
                display_name="Model",
                description="Model to use for text extraction",
                default="gpt-3.5-turbo",
                is_multi=False,
                options={
                    "gpt-3.5-turbo": {
                        "display_name": "GPT 3.5 Turbo",
                        "value": "gpt-3.5-turbo",
                    }
                },
            ),
            StringParameter(
                name="output_schema",
                display_name="Output JSON Schema",
                description="Schema used for the output",
                default="""[
    {
        "field_name": "string",
        "field_value": "string" or "array of string"
    }
]""",
            ),
            FloatParameter(
                name="rate_limit_per_minute",
                display_name="Rate Limit per Minute",
                description="Rate limit to use when calling openai API",
                default=3500,
                min=0.000001,
                hidden=True,
            ),
        ]

        return Parameters(
            name=Model.openai,
            display_name="OpenAI",
            parameters=parameters,
        )

    def _predict(self, data: Iterable[str]) -> Iterable[JSONType]:
        import time
        import json
        import openai
        from actableai.utils.openai import num_tokens_from_messages

        rate_limit_per_minute = self.parameters["rate_limit_per_minute"]
        delay = 60.0 / rate_limit_per_minute

        fields_to_extract = self.parameters["fields_to_extract"]
        model = self.parameters["model"]
        output_schema = self.parameters["output_schema"]

        extracted_data = []
        for document in data:
            prompt = f"""Extract the following fields from the provided document: {fields_to_extract}

Using this JSON format as a result:
{output_schema}

The JSON Object:
"""

            time.sleep(delay)
            messages = [
                {"role": "system", "content": document},
                {"role": "user", "content": prompt},
            ]

            num_tokens = num_tokens_from_messages(
                messages=messages,
                model=model,
            )
            max_tokens = 4096 - num_tokens - 1

            chat_completion_result = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )

            # TODO check if max tokens is reached

            content = chat_completion_result["choices"][0]["message"]["content"]

            try:
                extracted_fields = json.loads(content)
            except ValueError:
                extracted_fields = content

            extracted_data.append(extracted_fields)

        return extracted_data
