import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
from utils.logger import logger
from config import DATA_DIR


class IdentityRAG:

    def __init__(self):
        self.storage_path = DATA_DIR / "identity_profiles"
        self.profiles: Dict[str, Dict] = {}
        self._initialized = False

    def initialize(self):
        if self._initialized:
            logger.warning("Identity RAG already initialized")
            return

        logger.info(f"Initializing Identity RAG at {self.storage_path}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_profiles()
        self._initialized = True
        logger.success(f"Identity RAG initialized ({len(self.profiles)} profiles)")

    def _load_profiles(self):
        profiles_file = self.storage_path / "profiles.json"
        if profiles_file.exists():
            with open(profiles_file, 'r', encoding='utf-8') as f:
                self.profiles = json.load(f)

    def _save_profiles(self):
        profiles_file = self.storage_path / "profiles.json"
        with open(profiles_file, 'w', encoding='utf-8') as f:
            json.dump(self.profiles, f, indent=2, ensure_ascii=False)

    def get_profile(self, identity_id: str) -> Optional[Dict]:
        if not self._initialized:
            raise RuntimeError("Identity RAG not initialized")
        return self.profiles.get(identity_id)

    def create_profile(self, identity_id: str) -> Dict:
        if not self._initialized:
            raise RuntimeError("Identity RAG not initialized")

        if identity_id in self.profiles:
            return self.profiles[identity_id]

        now = datetime.now().isoformat()
        profile = {
            "identity_id": identity_id,
            "name": None,
            "age": None,
            "personality_brief": "",
            "interests": [],
            "relationship": "stranger",
            "first_met": now,
            "last_seen": now,
            "interaction_count": 1
        }

        self.profiles[identity_id] = profile
        self._save_profiles()
        logger.info(f"Created profile for: {identity_id}")
        return profile

    def get_or_create_profile(self, identity_id: str) -> Dict:
        existing = self.get_profile(identity_id)
        if existing:
            self.update_last_seen(identity_id)
            return existing
        return self.create_profile(identity_id)

    def update_profile(self, identity_id: str, field: str, value: Any) -> bool:
        if not self._initialized:
            raise RuntimeError("Identity RAG not initialized")

        allowed_fields = ["name", "age", "personality_brief", "interests", "relationship"]
        if field not in allowed_fields:
            logger.error(f"Invalid field: {field}. Allowed: {allowed_fields}")
            return False

        if identity_id not in self.profiles:
            logger.error(f"Identity not found: {identity_id}")
            return False

        self.profiles[identity_id][field] = value
        self._save_profiles()
        logger.info(f"Updated {identity_id}.{field} = {value}")
        return True

    def update_last_seen(self, identity_id: str) -> bool:
        if not self._initialized:
            raise RuntimeError("Identity RAG not initialized")

        if identity_id not in self.profiles:
            return False

        self.profiles[identity_id]["last_seen"] = datetime.now().isoformat()
        self.profiles[identity_id]["interaction_count"] = self.profiles[identity_id].get("interaction_count", 0) + 1
        self._save_profiles()
        return True

    def get_all_profiles(self) -> List[Dict]:
        if not self._initialized:
            raise RuntimeError("Identity RAG not initialized")
        return list(self.profiles.values())

    def delete_profile(self, identity_id: str) -> bool:
        if not self._initialized:
            raise RuntimeError("Identity RAG not initialized")

        if identity_id not in self.profiles:
            return False

        del self.profiles[identity_id]
        self._save_profiles()
        logger.info(f"Deleted profile: {identity_id}")
        return True

    def get_profiles_by_ids(self, identity_ids: List[str]) -> List[Dict]:
        if not self._initialized:
            raise RuntimeError("Identity RAG not initialized")
        
        result = []
        for identity_id in identity_ids:
            profile = self.get_or_create_profile(identity_id)
            result.append(profile)
        return result

