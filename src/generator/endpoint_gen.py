import os
import asyncio
import openai
from typing import Callable
from transformers import AutoTokenizer
from src.data import Hit, Result
from src.generator.base import BaseGenerator


class EndpointGenerator(BaseGenerator):
    """Generation via a running vLLM (or OpenAI-compatible) serving endpoint."""

    def __init__(
        self,
        model_name_or_path: str,
        api_key: str = 'EMPTY',
        base_url: str = 'http://localhost:8000/v1',
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ):
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.client = openai.OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY', api_key),
            base_url=base_url,
            max_retries=10,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def _format_prompt(self, result: Result, hits: list[Hit], prompt_builder: Callable) -> str:
        messages = prompt_builder(result.query, hits, result.meta)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        results: list[Result],
        contexts: dict[str, list[Hit]],
        prompt_builder: Callable,
    ) -> dict[str, str]:
        prompts = {
            r.qid: self._format_prompt(r, contexts.get(r.qid, []), prompt_builder)
            for r in results
        }
        return self.loop.run_until_complete(self._agenerate(prompts))

    async def _agenerate(self, prompts: dict[str, str]) -> dict[str, str]:
        def _get_output(prompt: str) -> str:
            response = self.client.completions.create(
                model=self.model_name_or_path,
                prompt=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].text

        outputs = await asyncio.gather(*[
            asyncio.to_thread(_get_output, prompt)
            for prompt in prompts.values()
        ])
        return dict(zip(prompts.keys(), outputs))
