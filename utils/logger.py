from datetime import datetime
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )


class Logger:

    @staticmethod
    def info(message, prefix="ℹ️"):
        print(f"{prefix} {message}")
    
    @staticmethod
    def success(message, prefix="✅"):
        print(f"{prefix} {message}")
    
    @staticmethod
    def warning(message, prefix="⚠️"):
        print(f"{prefix} {message}")
    
    @staticmethod
    def error(message, prefix="❌"):
        print(f"{prefix} {message}", file=sys.stderr)
    
    @staticmethod
    def debug(message, prefix="🔍"):
        print(f"{prefix} {message}")
    
    @staticmethod
    def recording(message, prefix="🎙️"):
        print(f"\n{prefix} {message}")
    
    @staticmethod
    def audio_event(message, prefix="🔊"):
        print(f"{prefix} {message}")
    
    @staticmethod
    def transcription(text, prefix="📝"):
        print(f"{prefix} Transcription: {text}")
    
    @staticmethod
    def assistant_response(text, prefix="🤖"):
        print(f"\n{prefix} Assistant: {text}")
    
    @staticmethod
    def metrics(tokens_per_second=None, generation_time=None, num_tokens=None):
        if tokens_per_second is not None:
            print(f"   ⚡ Speed: {tokens_per_second:.1f} tokens/s")
        if generation_time is not None:
            print(f"   ⏱️  Time: {generation_time:.2f}s")
        if num_tokens is not None:
            print(f"   📊 Tokens: {num_tokens}")
    
    @staticmethod
    def separator(char="─", length=60):
        print(char * length)
    
    @staticmethod
    def header(title):
        Logger.separator()
        print(f"  {title}")
        Logger.separator()
    
    @staticmethod
    def timestamp():
        return datetime.now().strftime('%H:%M:%S')
    
    @staticmethod
    def event_details(label, duration, confidence, chunks, filename):
        print(f"   Label: {label}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Confidence: {confidence:.0%}")
        print(f"   Chunks: {chunks}")
        print(f"   File: {filename}")


logger = Logger()

