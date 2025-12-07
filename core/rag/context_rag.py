import uuid
import time
import numpy as np
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, OrderBy, Direction
)
from typing import List, Dict, Optional
from utils.logger import logger
from config import QDRANT_CONTEXT_PATH, CONTEXT_COLLECTION


class ContextRAG:

    def __init__(self, embedding_dim: int = 384):
        self.client = None
        self.collection_name = CONTEXT_COLLECTION
        self.embedding_dim = embedding_dim
        self._initialized = False

    def initialize(self):
        if self._initialized:
            logger.warning("Context RAG already initialized")
            return

        logger.info(f"Initializing Context RAG at {QDRANT_CONTEXT_PATH}")
        self.client = QdrantClient(path=QDRANT_CONTEXT_PATH)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

        self._initialized = True
        logger.success("Context RAG initialized")

    def remember(
        self,
        identity_id: str,
        content: str,
        text_embedding: np.ndarray,
        memory_type: str = "fact",
        importance: float = 0.5,
        tags: List[str] = None
    ) -> Optional[str]:
        if not self._initialized:
            raise RuntimeError("Context RAG not initialized")

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = time.time()
        datetime_str = datetime.now().isoformat()

        payload = {
            "memory_id": memory_id,
            "identity_id": identity_id,
            "timestamp": now,
            "datetime_str": datetime_str,
            "memory_type": memory_type,
            "content": content,
            "importance": importance,
            "tags": tags or []
        }

        point_id = abs(hash(memory_id)) % (2**63 - 1)

        point = PointStruct(
            id=point_id,
            vector=text_embedding.tolist(),
            payload=payload
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        logger.info(f"Remembered for {identity_id}: {content[:50]}...")
        return memory_id

    def recall(
        self,
        identity_id: str,
        query_embedding: np.ndarray,
        limit: int = 5,
        memory_type: Optional[str] = None,
        score_threshold: float = 0.4
    ) -> List[Dict]:
        if not self._initialized:
            raise RuntimeError("Context RAG not initialized")

        conditions = [
            FieldCondition(
                key="identity_id",
                match=MatchValue(value=identity_id)
            )
        ]

        if memory_type:
            conditions.append(
                FieldCondition(
                    key="memory_type",
                    match=MatchValue(value=memory_type)
                )
            )

        query_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold
        )

        return [
            {
                "memory_id": hit.payload["memory_id"],
                "content": hit.payload["content"],
                "memory_type": hit.payload.get("memory_type", "fact"),
                "importance": hit.payload.get("importance", 0.5),
                "datetime_str": hit.payload.get("datetime_str", ""),
                "tags": hit.payload.get("tags", []),
                "score": hit.score
            }
            for hit in results.points
        ]

    def recall_recent(
        self,
        identity_id: str,
        limit: int = 5
    ) -> List[Dict]:
        if not self._initialized:
            raise RuntimeError("Context RAG not initialized")

        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="identity_id",
                        match=MatchValue(value=identity_id)
                    )
                ]
            ),
            limit=100
        )

        memories = [
            {
                "memory_id": point.payload["memory_id"],
                "content": point.payload["content"],
                "memory_type": point.payload.get("memory_type", "fact"),
                "importance": point.payload.get("importance", 0.5),
                "timestamp": point.payload.get("timestamp", 0),
                "datetime_str": point.payload.get("datetime_str", ""),
                "tags": point.payload.get("tags", [])
            }
            for point in scroll_result[0]
        ]

        memories.sort(key=lambda m: m["timestamp"], reverse=True)
        return memories[:limit]

    def get_memories_by_identity(
        self,
        identity_id: str,
        limit: int = 100
    ) -> List[Dict]:
        if not self._initialized:
            raise RuntimeError("Context RAG not initialized")

        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="identity_id",
                        match=MatchValue(value=identity_id)
                    )
                ]
            ),
            limit=limit
        )

        return [
            {
                "memory_id": point.payload["memory_id"],
                "content": point.payload["content"],
                "memory_type": point.payload.get("memory_type", "fact"),
                "importance": point.payload.get("importance", 0.5),
                "timestamp": point.payload.get("timestamp", 0),
                "datetime_str": point.payload.get("datetime_str", ""),
                "tags": point.payload.get("tags", [])
            }
            for point in scroll_result[0]
        ]

    def get_memory_count(self, identity_id: str) -> int:
        if not self._initialized:
            raise RuntimeError("Context RAG not initialized")

        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="identity_id",
                        match=MatchValue(value=identity_id)
                    )
                ]
            ),
            limit=1000
        )

        return len(scroll_result[0])

