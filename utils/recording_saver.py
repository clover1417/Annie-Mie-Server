import os
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from utils.logger import logger


class RecordingSaver:
    def __init__(self, base_dir: str = "server/data/recordings"):
        self.base_dir = Path(base_dir)
        self.current_session_dir: Optional[Path] = None
        self.current_frames_dir: Optional[Path] = None

    def create_session_folder(self, session_name: str) -> Path:
        session_dir = self.base_dir / session_name
        frames_dir = session_dir / "frames"

        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        self.current_session_dir = session_dir
        self.current_frames_dir = frames_dir

        logger.info(f"Created server session folder: {session_name}")
        return session_dir

    def save_audio_from_base64(self, audio_base64: str, audio_format: str = "wav") -> str:
        if not self.current_session_dir:
            raise RuntimeError("Session folder not created")

        ext = audio_format if audio_format in ("wav", "flac") else "wav"
        dest_path = self.current_session_dir / f"audio.{ext}"

        try:
            audio_bytes = base64.b64decode(audio_base64)
            with open(dest_path, "wb") as f:
                f.write(audio_bytes)

            logger.info(f"Saved audio: {dest_path}")
            return str(dest_path)
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def save_frame_from_base64(self, frame_base64: str, frame_number: int) -> str:
        if not self.current_frames_dir:
            raise RuntimeError("Frames folder not created")

        filename = f"frame_{frame_number:04d}.jpg"
        dest_path = self.current_frames_dir / filename

        try:
            img_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imwrite(str(dest_path), frame)
            return str(dest_path)
        except Exception as e:
            logger.error(f"Failed to save frame {frame_number}: {e}")
            return ""

    def get_current_session_info(self) -> dict:
        if not self.current_session_dir:
            return {"session_active": False}

        return {
            "session_active": True,
            "session_dir": str(self.current_session_dir),
            "frames_dir": str(self.current_frames_dir),
            "session_name": self.current_session_dir.name,
        }

