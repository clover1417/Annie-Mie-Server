import os
import sys

if sys.platform == "win32":
    from config import FFMPEG_PATH
    if os.path.exists(FFMPEG_PATH):
        os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(FFMPEG_PATH)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeForConditionalGeneration, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
from utils.logger import logger
from qwen_omni_utils import process_mm_info
import torch
import time
from typing import Optional

from config import (
    QWEN_MODEL_ID,
    QWEN_DEVICE,
    QWEN_QUANTIZATION,
    QWEN_MAX_TOKENS,
    QWEN_TEMPERATURE,
    QWEN_TOP_P,
    QWEN_TOP_K,
    QWEN_DO_SAMPLE,
    SYSTEM_PROMPTS,
)


class QwenSession:

    def __init__(self, model_id: str = None):
        self.model_id = model_id or QWEN_MODEL_ID
        self.device = QWEN_DEVICE
        self.processor = None
        self.model = None
        self.base_system_instruction = None

    def initialize(self):
        logger.info("Loading base system instruction...")
        if not os.path.exists(SYSTEM_PROMPTS):
            raise FileNotFoundError(f"System prompt file not found: {SYSTEM_PROMPTS}")

        with open(SYSTEM_PROMPTS, 'r', encoding='utf-8') as f:
            self.base_system_instruction = f.read().strip()

        if not self.base_system_instruction:
            raise ValueError("System prompt file is empty!")

        logger.success(f"Base system instruction loaded ({len(self.base_system_instruction)} chars)")

        # CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Set default CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            # Clear any cached memory before loading
            torch.cuda.empty_cache()

        bnb_config = None
        model_dtype = torch.bfloat16
        use_device_map = "auto"
        
        if QWEN_QUANTIZATION == "4bit":
            logger.info("Setting up 4-bit quantization config...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif QWEN_QUANTIZATION == "8bit":
            logger.info("Setting up 8-bit quantization config...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        elif QWEN_QUANTIZATION == "none" or QWEN_QUANTIZATION is None or QWEN_QUANTIZATION == "":
            # No quantization - full precision mode for maximum speed
            # Requires significant VRAM (~60GB+ for 30B model)
            logger.info("Setting up NO quantization mode (full bfloat16 precision)...")
            logger.info("This requires ~60GB+ VRAM but provides fastest inference")
            model_dtype = torch.bfloat16
            # Use single GPU device directly to avoid device_map overhead
            use_device_map = "cuda:0"

        logger.info("Loading processor...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        logger.info(f"Loading model with {QWEN_QUANTIZATION or 'none'} quantization...")
        
        # Build model kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",  # Use flash attention for speed
        }
        
        if bnb_config is not None:
            # Quantized mode
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            # No quantization mode - load directly to GPU with full precision
            model_kwargs["torch_dtype"] = model_dtype
            model_kwargs["device_map"] = use_device_map
            # Low CPU memory usage during loading
            model_kwargs["low_cpu_mem_usage"] = True
        
        try:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )
        except Exception as e:
            # Fallback if flash_attention_2 is not available
            logger.warning(f"Flash Attention 2 not available: {e}")
            logger.info("Falling back to SDPA attention...")
            model_kwargs["attn_implementation"] = "sdpa"
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )

        if hasattr(self.model.config, 'return_audio'):
            self.model.config.return_audio = False

        torch.set_float32_matmul_precision("medium")

        # Skip torch.compile for no-quantization mode - it adds latency on first inference
        if QWEN_QUANTIZATION not in ["none", None, ""]:
            try:
                logger.info("Attempting torch.compile optimization...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.success("Model compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")
                logger.info("Continuing without compilation...")
        else:
            logger.info("Skipping torch.compile for no-quantization mode (faster first token)")

        self.model.eval()
        
        # Log VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        logger.success(f"Model loaded on device: {self.model.device}")

    def build_system_instruction(self, identity_context: Optional[str] = None) -> str:
        if not identity_context:
            return self.base_system_instruction
        
        return f"{self.base_system_instruction}\n\n---\nCURRENT SPEAKERS:{identity_context}"

    def generate(
        self,
        conversation: list,
        token_callback=None,
        stats_callback=None,
        identity_context: Optional[str] = None
    ) -> str:
        USE_AUDIO_IN_VIDEO = True

        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        system_instruction = self.build_system_instruction(identity_context)
        if system_instruction:
            text = f"System: {system_instruction}\n\n{text}"

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=QWEN_MAX_TOKENS,
            temperature=QWEN_TEMPERATURE,
            top_p=QWEN_TOP_P,
            top_k=QWEN_TOP_K,
            do_sample=QWEN_DO_SAMPLE,
            streamer=streamer,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            return_audio=False
        )

        start_time = time.time()

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        debounce = False
        generated_text = ""

        try:
            for new_text in streamer:
                if not debounce:
                    debounce = True
                    first_token_time = time.time() - start_time
                    if stats_callback:
                        stats_callback("first_token", first_token_time)

                generated_text += new_text

                if token_callback:
                    token_callback(new_text)

        except Exception as e:
            logger.error(f"Generation error: {e}")
            if stats_callback:
                stats_callback("error", str(e))
            thread.join()
            raise

        thread.join()

        token_count = len(self.processor.tokenizer.encode(generated_text, add_special_tokens=False))
        elapsed_time = time.time() - start_time
        tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0

        if stats_callback:
            stats_callback("complete", {
                "tokens": token_count,
                "time": elapsed_time,
                "tok_per_sec": tokens_per_second
            })

        return generated_text.strip()

