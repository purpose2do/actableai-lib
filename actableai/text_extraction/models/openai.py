from functools import lru_cache
from typing import Iterable, List

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

    def _open_ai_completion(
        self,
        fields_to_extract: List[str],
        output_schema: str,
        document: str,
        model: str,
    ) -> str:
        import openai
        from actableai.utils.openai import num_tokens_from_messages

        prompt = f"""Extract the following fields from the provided document: {fields_to_extract}

Using this JSON format as a result:
{output_schema}

The JSON Object:
"""

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

        return chat_completion_result["choices"][0]["message"]["content"]

    def _predict(self, data: Iterable[str]) -> Iterable[JSONType]:
        import time
        import openai
        import backoff

        _open_ai_completion_func = backoff.on_exception(
            backoff.expo, openai.error.RateLimitError
        )(self._open_ai_completion)

        rate_limit_per_minute = self.parameters["rate_limit_per_minute"]
        delay = 60.0 / rate_limit_per_minute

        fields_to_extract = self.parameters["fields_to_extract"]
        model = self.parameters["model"]
        output_schema = self.parameters["output_schema"]

        extracted_data = []
        for document in data:
            time.sleep(delay)
            content = _open_ai_completion_func(
                fields_to_extract=fields_to_extract,
                output_schema=output_schema,
                document=document,
                model=model,
            )
            extracted_data.append(content)

        return extracted_data
