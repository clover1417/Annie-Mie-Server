"""
vLLM-based inference for Qwen3-Omni model.
Uses AsyncLLMEngine for streaming text generation.
"""

import os
import sys
import asyncio

if sys.platform == "win32":
    from config import FFMPEG_PATH
    if os.path.exists(FFMPEG_PATH):
        os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(FFMPEG_PATH)

# vLLM engine v1 not supported for Qwen3-Omni
os.environ['VLLM_USE_V1'] = '0'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import time
from typing import Optional, Callable
from threading import Thread

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind

from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from utils.logger import logger

from config import (
    QWEN_MODEL_ID,
    QWEN_MAX_TOKENS,
    QWEN_TEMPERATURE,
    QWEN_TOP_P,
    QWEN_TOP_K,
    QWEN_DO_SAMPLE,
    SYSTEM_PROMPTS,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_NUM_SEQS,
    VLLM_LIMIT_MM_PER_PROMPT,
)


class QwenSession:
    """
    vLLM-based Qwen3-Omni inference session.
    Provides streaming text generation with multimodal support.
    """

    def __init__(self, model_id: str = None):
        self.model_id = model_id or QWEN_MODEL_ID
        self.processor = None
        self.engine: Optional[AsyncLLMEngine] = None
        self.llm: Optional[LLM] = None
        self.base_system_instruction = None
        self._request_counter = 0
        self._use_async_engine = True  # Set to True for streaming support

    def initialize(self):
        """Initialize the vLLM engine and processor."""
        logger.info("Loading base system instruction...")
        if not os.path.exists(SYSTEM_PROMPTS):
            raise FileNotFoundError(f"System prompt file not found: {SYSTEM_PROMPTS}")

        with open(SYSTEM_PROMPTS, 'r', encoding='utf-8') as f:
            self.base_system_instruction = f.read().strip()

        if not self.base_system_instruction:
            raise ValueError("System prompt file is empty!")

        logger.success(f"Base system instruction loaded ({len(self.base_system_instruction)} chars)")

        # CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()

        # Load processor for chat template
        logger.info("Loading Qwen3OmniMoeProcessor...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        logger.success("Processor loaded")

        # vLLM engine configuration
        tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        logger.info(f"Initializing vLLM with {tensor_parallel_size} GPU(s)...")
        logger.info(f"  Model: {self.model_id}")
        logger.info(f"  GPU Memory Utilization: {VLLM_GPU_MEMORY_UTILIZATION}")
        logger.info(f"  Max Model Len: {VLLM_MAX_MODEL_LEN}")
        logger.info(f"  Max Num Seqs: {VLLM_MAX_NUM_SEQS}")
        
        if self._use_async_engine:
            # Use AsyncLLMEngine for streaming support
            engine_args = AsyncEngineArgs(
                model=self.model_id,
                trust_remote_code=True,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                tensor_parallel_size=tensor_parallel_size,
                max_num_seqs=VLLM_MAX_NUM_SEQS,
                max_model_len=VLLM_MAX_MODEL_LEN,
                limit_mm_per_prompt={
                    'image': VLLM_LIMIT_MM_PER_PROMPT,
                    'video': VLLM_LIMIT_MM_PER_PROMPT,
                    'audio': VLLM_LIMIT_MM_PER_PROMPT,
                },
                seed=1234,
                dtype="bfloat16",
                enforce_eager=False,  # Use CUDA graphs for faster inference on A100
                enable_chunked_prefill=True,  # Better latency for long prompts
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.success("AsyncLLMEngine initialized for streaming")
        else:
            # Fallback to sync LLM (no streaming)
            self.llm = LLM(
                model=self.model_id,
                trust_remote_code=True,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                tensor_parallel_size=tensor_parallel_size,
                limit_mm_per_prompt={
                    'image': VLLM_LIMIT_MM_PER_PROMPT,
                    'video': VLLM_LIMIT_MM_PER_PROMPT,
                    'audio': VLLM_LIMIT_MM_PER_PROMPT,
                },
                max_num_seqs=VLLM_MAX_NUM_SEQS,
                max_model_len=VLLM_MAX_MODEL_LEN,
                seed=1234,
            )
            logger.success("LLM engine initialized (non-streaming)")

        # Log VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        logger.success("vLLM model loaded successfully")

    def build_system_instruction(self, identity_context: Optional[str] = None) -> str:
        """Build the complete system instruction with optional identity context."""
        if not identity_context:
            return self.base_system_instruction
        
        return f"{self.base_system_instruction}\n\n---\nCURRENT SPEAKERS:{identity_context}"

    def _strip_thinking_tags(self, text: str) -> str:
        """Remove Qwen3 <think>...</think> tags from generated text."""
        import re
        # Remove <think>...</think> blocks (including multiline)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Clean up any extra whitespace left behind
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def _prepare_inputs(self, conversation: list, identity_context: Optional[str] = None) -> dict:
        """Prepare inputs for vLLM generation."""
        USE_AUDIO_IN_VIDEO = True
        
        # Process multimodal data
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
            # Note: Don't use enable_thinking=False as it adds empty <think></think> tags
            # which causes 0 token generation. Better to filter thinking in post-processing.
        )
        
        # Prepend system instruction
        system_instruction = self.build_system_instruction(identity_context)
        if system_instruction:
            text = f"System: {system_instruction}\n\n{text}"
        
        # Build vLLM input format
        inputs = {
            'prompt': text,
            'multi_modal_data': {},
            'mm_processor_kwargs': {
                'use_audio_in_video': USE_AUDIO_IN_VIDEO,
            },
        }
        
        if images is not None:
            inputs['multi_modal_data']['image'] = images
        if videos is not None:
            inputs['multi_modal_data']['video'] = videos
        if audios is not None:
            inputs['multi_modal_data']['audio'] = audios
            
        return inputs

    def _get_sampling_params(self, stream: bool = False) -> SamplingParams:
        """Get sampling parameters for generation."""
        return SamplingParams(
            temperature=QWEN_TEMPERATURE if QWEN_DO_SAMPLE else 0.0,
            top_p=QWEN_TOP_P,
            top_k=QWEN_TOP_K,
            max_tokens=QWEN_MAX_TOKENS,
            # Use DELTA for streaming to get incremental tokens
            output_kind=RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY,
        )

    async def generate_async(
        self,
        conversation: list,
        token_callback: Callable[[str], None] = None,
        stats_callback: Callable[[str, any], None] = None,
        identity_context: Optional[str] = None
    ) -> str:
        """
        Async generate response with streaming support.
        
        Args:
            conversation: List of conversation messages
            token_callback: Callback for each generated token
            stats_callback: Callback for generation statistics
            identity_context: Optional identity context string
            
        Returns:
            Complete generated text
        """
        # Prepare inputs
        inputs = self._prepare_inputs(conversation, identity_context)
        
        start_time = time.time()
        
        if self._use_async_engine and self.engine is not None:
            # Use async engine with streaming
            generated_text = await self._async_generate_streaming(
                inputs=inputs,
                token_callback=token_callback,
                stats_callback=stats_callback,
                start_time=start_time,
            )
        else:
            # Fallback to sync generation (run in executor to not block)
            loop = asyncio.get_event_loop()
            generated_text = await loop.run_in_executor(
                None,
                lambda: self._generate_sync(
                    inputs=inputs,
                    token_callback=token_callback,
                    stats_callback=stats_callback,
                    start_time=start_time,
                )
            )
        
        # Strip Qwen3 thinking tags from response
        generated_text = self._strip_thinking_tags(generated_text)
        
        return generated_text.strip()

    def generate(
        self,
        conversation: list,
        token_callback: Callable[[str], None] = None,
        stats_callback: Callable[[str, any], None] = None,
        identity_context: Optional[str] = None
    ) -> str:
        """Sync wrapper for backward compatibility. Prefer generate_async."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If called from async context, create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.generate_async(conversation, token_callback, stats_callback, identity_context)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                self.generate_async(conversation, token_callback, stats_callback, identity_context)
            )

    async def _async_generate_streaming(
        self,
        inputs: dict,
        token_callback: Callable[[str], None],
        stats_callback: Callable[[str, any], None],
        start_time: float,
    ) -> str:
        """Async streaming generation."""
        self._request_counter += 1
        request_id = f"request-{self._request_counter}"
        
        sampling_params = self._get_sampling_params(stream=True)
        
        generated_text = ""
        first_token_sent = False
        token_count = 0
        
        try:
            # Build prompt dict for vLLM - multimodal data goes inside prompt
            prompt_input = {
                'prompt': inputs['prompt'],
            }
            
            # Add multimodal data if present
            if inputs.get('multi_modal_data'):
                prompt_input['multi_modal_data'] = inputs['multi_modal_data']
            if inputs.get('mm_processor_kwargs'):
                prompt_input['mm_processor_kwargs'] = inputs['mm_processor_kwargs']
            
            # Start generation
            results_generator = self.engine.generate(
                prompt=prompt_input,
                sampling_params=sampling_params,
                request_id=request_id,
            )
            
            async for request_output in results_generator:
                if request_output.outputs:
                    for output in request_output.outputs:
                        # Get delta text (incremental output)
                        delta = output.text
                        
                        if delta:
                            if not first_token_sent:
                                first_token_sent = True
                                first_token_time = time.time() - start_time
                                logger.info(f"⚡ First token: {first_token_time:.2f}s")
                                if stats_callback:
                                    stats_callback("first_token", first_token_time)
                            
                            generated_text += delta
                            token_count += 1
                            
                            if token_callback:
                                token_callback(delta)
            
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            if stats_callback:
                stats_callback("error", str(e))
            raise
        
        # Send completion stats
        elapsed_time = time.time() - start_time
        tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
        
        logger.success(f"📊 Generated {token_count} tokens in {elapsed_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        if stats_callback:
            stats_callback("complete", {
                "tokens": token_count,
                "time": elapsed_time,
                "tok_per_sec": tokens_per_second
            })
        
        return generated_text

    def _generate_sync(
        self,
        inputs: dict,
        token_callback: Callable[[str], None],
        stats_callback: Callable[[str, any], None],
        start_time: float,
    ) -> str:
        """Sync generation fallback (no true streaming)."""
        sampling_params = self._get_sampling_params(stream=False)
        
        # Generate all at once
        outputs = self.llm.generate([inputs], sampling_params=sampling_params)
        
        generated_text = outputs[0].outputs[0].text if outputs else ""
        
        # Report first token immediately (since we got all at once)
        if stats_callback:
            stats_callback("first_token", time.time() - start_time)
        
        # Send all tokens at once via callback
        if token_callback and generated_text:
            token_callback(generated_text)
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        token_count = len(self.processor.tokenizer.encode(generated_text, add_special_tokens=False))
        tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
        
        if stats_callback:
            stats_callback("complete", {
                "tokens": token_count,
                "time": elapsed_time,
                "tok_per_sec": tokens_per_second
            })
        
        return generated_text
