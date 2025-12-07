import json
import os
from typing import List, Dict, Optional
from utils.logger import logger
from config import CONVERSATION_LOG


class ConversationManager:

    def __init__(self):
        self.log_file = CONVERSATION_LOG
        self.current_identity_context: Optional[str] = None
        self._initialized = False

    def load(self):
        if self._initialized:
            return
        
        self._reload_from_file()
        self._initialized = True

    def _reload_from_file(self) -> List[Dict]:
        try:
            if not os.path.exists(self.log_file):
                return []

            with open(self.log_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            return history

        except Exception as e:
            logger.error(f"Could not load conversation log: {e}")
            return []

    def _save_to_file(self, history: List[Dict]):
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save conversation log: {e}")

    def set_identity_context(self, identity_contexts: List[Dict]):
        if not identity_contexts:
            self.current_identity_context = None
            return

        context_parts = []
        for ctx in identity_contexts:
            identity_id = ctx.get("identity_id", "unknown")
            name = ctx.get("name")
            age = ctx.get("age")
            relationship = ctx.get("relationship", "stranger")
            personality = ctx.get("personality_brief", "")
            interests = ctx.get("interests", [])
            interaction_count = ctx.get("interaction_count", 0)
            is_first = ctx.get("is_first_meeting", False)

            person_info = f"\n[{identity_id}]"
            if name:
                person_info += f"\n- Name: {name}"
            else:
                person_info += "\n- Name: Unknown"
            
            if age:
                person_info += f"\n- Age: {age}"
            
            person_info += f"\n- Relationship: {relationship}"
            
            if personality:
                person_info += f"\n- Personality: {personality}"
            
            if interests:
                person_info += f"\n- Interests: {', '.join(interests)}"
            
            if is_first:
                person_info += "\n- Status: First meeting!"
            elif interaction_count > 1:
                person_info += f"\n- Met {interaction_count} times"

            recent_memories = ctx.get("recent_memories", [])
            if recent_memories:
                person_info += "\n- Recent memories:"
                for mem in recent_memories[:2]:
                    content = mem.get("content", "")
                    mem_type = mem.get("memory_type", "fact")
                    person_info += f"\n  [{mem_type}] {content}"

            context_parts.append(person_info)

        self.current_identity_context = "\n".join(context_parts)

    def get_identity_context(self) -> Optional[str]:
        return self.current_identity_context

    def add_assistant_response(self, response_text: str):
        history = self._reload_from_file()
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        })
        self._save_to_file(history)

    def add_system_note(self, note_text: str):
        history = self._reload_from_file()
        history.append({
            "role": "user",
            "content": [{"type": "text", "text": f"[System Note] {note_text}"}]
        })
        self._save_to_file(history)

    def process_audio(
        self,
        audio_path: str,
        session_folder: str = None,
        identity_ids: List[str] = None
    ):
        logger.info(f"Adding audio to history: {os.path.basename(audio_path)}")

        content = []

        if identity_ids:
            ids_text = ", ".join(identity_ids)
            content.append({
                "type": "text",
                "text": f"Speaker(s): {ids_text}"
            })
        else:
            content.append({
                "type": "text",
                "text": "Speaker: Unknown"
            })

        content.append({
            "type": "audio",
            "audio": audio_path,
        })

        if session_folder:
            frames_dir = os.path.join(session_folder, "frames")
            if os.path.exists(frames_dir):
                frame_files = sorted([
                    os.path.join(frames_dir, f)
                    for f in os.listdir(frames_dir)
                    if f.endswith(('.jpg', '.jpeg', '.png'))
                ])

                for frame_path in frame_files:
                    content.append({
                        "type": "image",
                        "image": frame_path
                    })

                if frame_files:
                    logger.info(f"Added {len(frame_files)} frame references to conversation")

        history = self._reload_from_file()
        history.append({
            "role": "user",
            "content": content
        })
        self._save_to_file(history)

    def get_history(self) -> List[Dict]:
        return self._reload_from_file()

    def clear_history(self):
        self._save_to_file([])
        self.current_identity_context = None
        logger.info("Conversation history cleared")
