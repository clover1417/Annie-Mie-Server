from typing import Dict, Any, Optional
from utils.logger import logger
from core.rag.rag_manager import RAGManager


class FunctionExecutor:

    def __init__(self, rag_manager: RAGManager):
        self.rag_manager = rag_manager
        self.current_identity_id: Optional[str] = None

    def set_identity(self, identity_id: str):
        self.current_identity_id = identity_id

    def execute(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self.current_identity_id:
            return {
                "success": False,
                "error": "No identity set for function execution"
            }

        try:
            if function_name == "remember":
                return self._execute_remember(arguments)
            elif function_name == "recall":
                return self._execute_recall(arguments)
            elif function_name == "recall_recent":
                return self._execute_recall_recent(arguments)
            elif function_name == "update_user_profile":
                return self._execute_update_profile(arguments)
            elif function_name == "get_user_profile":
                return self._execute_get_profile(arguments)
            else:
                return {
                    "success": False,
                    "error": f"Unknown function: {function_name}"
                }
        except Exception as e:
            logger.error(f"Function execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _execute_remember(self, args: Dict) -> Dict:
        content = args.get("content")
        memory_type = args.get("memory_type", "fact")
        importance = args.get("importance", 0.5)

        if not content:
            return {"success": False, "error": "Content is required"}

        memory_id = self.rag_manager.remember(
            identity_id=self.current_identity_id,
            content=content,
            memory_type=memory_type,
            importance=float(importance)
        )

        if memory_id:
            logger.info(f"Remembered [{memory_type}]: {content[:50]}...")
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Remembered: {content}"
            }
        return {"success": False, "error": "Failed to store memory"}

    def _execute_recall(self, args: Dict) -> Dict:
        query = args.get("query")
        memory_type = args.get("memory_type")
        limit = args.get("limit", 5)

        if not query:
            return {"success": False, "error": "Query is required"}

        memories = self.rag_manager.recall(
            identity_id=self.current_identity_id,
            query=query,
            limit=int(limit),
            memory_type=memory_type
        )

        logger.info(f"Recalled {len(memories)} memories for query: {query[:30]}...")
        return {
            "success": True,
            "memories": memories,
            "count": len(memories)
        }

    def _execute_recall_recent(self, args: Dict) -> Dict:
        limit = args.get("limit", 5)

        memories = self.rag_manager.recall_recent(
            identity_id=self.current_identity_id,
            limit=int(limit)
        )

        logger.info(f"Retrieved {len(memories)} recent memories")
        return {
            "success": True,
            "memories": memories,
            "count": len(memories)
        }

    def _execute_update_profile(self, args: Dict) -> Dict:
        field = args.get("field")
        value = args.get("value")

        if not field or value is None:
            return {"success": False, "error": "Field and value are required"}

        if field == "interests" and isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]

        if field == "age":
            try:
                value = int(value)
            except ValueError:
                return {"success": False, "error": "Age must be a number"}

        success = self.rag_manager.update_profile(
            identity_id=self.current_identity_id,
            field=field,
            value=value
        )

        if success:
            logger.info(f"Updated profile {field} = {value}")
            return {"success": True, "message": f"Updated {field}"}
        return {"success": False, "error": f"Failed to update {field}"}

    def _execute_get_profile(self, args: Dict) -> Dict:
        profile = self.rag_manager.get_profile(self.current_identity_id)

        if profile:
            return {
                "success": True,
                "profile": profile
            }
        return {"success": False, "error": "Profile not found"}

