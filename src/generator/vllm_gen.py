import uuid
import asyncio
from typing import Callable, Optional
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
from src.data import Hit, Result
from src.generator.base import BaseGenerator
from src.generator.prompt import get_prompt_builder


class VLLMGenerator(BaseGenerator):

    def __init__(
        self,
        model_name_or_path: str,
        track: str = "rag",               # selects default prompt builder
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        num_gpus: int = 1,
        max_model_len: int = 10240,
        prompt_builder: Optional[Callable] = None,  # override per-track default
    ):
        args = AsyncEngineArgs(
            model=model_name_or_path,
            dtype=dtype,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.prompt_builder = prompt_builder or get_prompt_builder(track)

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def _format_prompt(self, result: Result, hits: list[Hit]) -> str:
        messages = self.prompt_builder(result.query, hits, result.meta)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, results: list[Result], contexts: dict[str, list[Hit]]) -> dict[str, str]:
        prompts = {
            r.qid: self._format_prompt(r, contexts.get(r.qid, []))
            for r in results
        }
        return self.loop.run_until_complete(self._agenerate(prompts))

    async def _agenerate(self, prompts: dict[str, str]) -> dict[str, str]:
        request_ids = {qid: str(uuid.uuid4()) for qid in prompts}
        iterators = {
            qid: await self.engine.add_request(request_ids[qid], prompt, self.sampling_params)
            for qid, prompt in prompts.items()
        }
        outputs = await asyncio.gather(*[
            self._collect(qid, it) for qid, it in iterators.items()
        ])
        return dict(outputs)

    async def _collect(self, qid: str, output_iterator) -> tuple[str, str]:
        self.engine._run_output_handler()
        finished = False
        text = ""
        while not finished:
            response = output_iterator.get_nowait() or await output_iterator.get()
            finished = response.finished
            text = response.outputs[0].text
        return qid, text
