from typing import List, Dict, Optional
from core.rag.identity_rag import IdentityRAG
from core.rag.context_rag import ContextRAG
from core.rag.embedding_models import EmbeddingModels
from utils.logger import logger


class RAGManager:

    def __init__(self):
        self.identity_rag = IdentityRAG()
        self.context_rag = ContextRAG(embedding_dim=384)
        self.embedding_models = EmbeddingModels()
        self._initialized = False

    def initialize(self):
        if self._initialized:
            logger.warning("RAG Manager already initialized")
            return

        logger.info("Initializing RAG Manager...")

        self.embedding_models.initialize()
        self.identity_rag.initialize()
        self.context_rag.initialize()

        self._initialized = True
        logger.success("RAG Manager initialized")

    def get_profile(self, identity_id: str) -> Optional[Dict]:
        return self.identity_rag.get_profile(identity_id)

    def get_or_create_profile(self, identity_id: str) -> Dict:
        return self.identity_rag.get_or_create_profile(identity_id)

    def get_profiles_by_ids(self, identity_ids: List[str]) -> List[Dict]:
        return self.identity_rag.get_profiles_by_ids(identity_ids)

    def update_profile(self, identity_id: str, field: str, value) -> bool:
        return self.identity_rag.update_profile(identity_id, field, value)

    def remember(
        self,
        identity_id: str,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        tags: List[str] = None
    ) -> Optional[str]:
        text_embedding = self.embedding_models.embed_text(content)

        return self.context_rag.remember(
            identity_id=identity_id,
            content=content,
            text_embedding=text_embedding,
            memory_type=memory_type,
            importance=importance,
            tags=tags
        )

    def recall(
        self,
        identity_id: str,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None
    ) -> List[Dict]:
        query_embedding = self.embedding_models.embed_text(query)

        return self.context_rag.recall(
            identity_id=identity_id,
            query_embedding=query_embedding,
            limit=limit,
            memory_type=memory_type
        )

    def recall_recent(
        self,
        identity_id: str,
        limit: int = 5
    ) -> List[Dict]:
        return self.context_rag.recall_recent(identity_id, limit)

    def get_identity_summary(self, identity_id: str) -> Dict:
        profile = self.identity_rag.get_profile(identity_id)
        memory_count = self.context_rag.get_memory_count(identity_id)
        recent = self.context_rag.recall_recent(identity_id, limit=3)

        return {
            "identity_id": identity_id,
            "profile": profile,
            "total_memories": memory_count,
            "recent_memories": recent
        }

    def build_identity_context(self, identity_id: str, profile: Dict) -> Dict:
        name = profile.get("name")
        age = profile.get("age")
        personality = profile.get("personality_brief", "")
        interests = profile.get("interests", [])
        relationship = profile.get("relationship", "stranger")
        interaction_count = profile.get("interaction_count", 0)

        return {
            "identity_id": identity_id,
            "name": name,
            "age": age,
            "personality_brief": personality,
            "interests": interests,
            "relationship": relationship,
            "interaction_count": interaction_count,
            "is_first_meeting": interaction_count <= 1
        }

    def build_multi_identity_context(self, identity_ids: List[str]) -> List[Dict]:
        contexts = []
        for identity_id in identity_ids:
            profile = self.get_or_create_profile(identity_id)
            context = self.build_identity_context(identity_id, profile)
            recent_memories = self.recall_recent(identity_id, limit=2)
            if recent_memories:
                context["recent_memories"] = recent_memories
            contexts.append(context)
        return contexts

