from typing import Dict, List

FUNCTION_DEFINITIONS: List[Dict] = [
    {
        "name": "remember",
        "description": "Store information about the user in long-term memory. Use this when learning new facts about the user (name, preferences, interests, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember about the user"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["fact", "preference", "event", "relationship"],
                    "description": "Type of memory: fact (general info), preference (likes/dislikes), event (something that happened), relationship (how user relates to Annie Mie)"
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score from 0.0 to 1.0 (default 0.5)"
                }
            },
            "required": ["content", "memory_type"]
        }
    },
    {
        "name": "recall",
        "description": "Search long-term memory for information about the user. Use when you need to remember something specific.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["fact", "preference", "event", "relationship"],
                    "description": "Optional: filter by memory type"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to retrieve (default 5)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "recall_recent",
        "description": "Get the most recent memories about the user, sorted by time.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent memories to retrieve (default 5)"
                }
            },
            "required": []
        }
    },
    {
        "name": "update_user_profile",
        "description": "Update the user's profile information.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["name", "age", "personality_brief", "interests", "relationship"],
                    "description": "Which field to update"
                },
                "value": {
                    "type": "string",
                    "description": "The new value (for interests, use comma-separated list)"
                }
            },
            "required": ["field", "value"]
        }
    },
    {
        "name": "get_user_profile",
        "description": "Get the current user's profile information.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


def get_function_schema() -> List[Dict]:
    return FUNCTION_DEFINITIONS


def get_function_by_name(name: str) -> Dict:
    for func in FUNCTION_DEFINITIONS:
        if func["name"] == name:
            return func
    return None

