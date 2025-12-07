import re
import json
from typing import Dict, List, Optional, Tuple, Any
from utils.logger import logger
from core.functions.definitions import get_function_by_name


class FunctionHandler:

    FUNCTION_PATTERNS = [
        r'<function_call>\s*(\{.*?\})\s*</function_call>',
        r'\[FUNCTION_CALL:\s*(\w+)\s*\((.*?)\)\]',
        r'```function\s*\n(\{.*?\})\s*\n```',
    ]

    @staticmethod
    def parse_function_calls(text: str) -> List[Dict[str, Any]]:
        function_calls = []

        json_pattern = r'<function_call>\s*(\{.*?\})\s*</function_call>'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                if "name" in call_data:
                    function_calls.append({
                        "name": call_data.get("name"),
                        "arguments": call_data.get("arguments", {}),
                        "raw": match.group(0)
                    })
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse function call JSON: {match.group(1)}")

        bracket_pattern = r'\[FUNCTION_CALL:\s*(\w+)\s*\((.*?)\)\]'
        for match in re.finditer(bracket_pattern, text, re.DOTALL):
            func_name = match.group(1)
            args_str = match.group(2).strip()
            try:
                args = json.loads(args_str) if args_str else {}
                function_calls.append({
                    "name": func_name,
                    "arguments": args,
                    "raw": match.group(0)
                })
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse function arguments: {args_str}")

        code_pattern = r'```function\s*\n(\{.*?\})\s*\n```'
        for match in re.finditer(code_pattern, text, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                if "name" in call_data:
                    function_calls.append({
                        "name": call_data.get("name"),
                        "arguments": call_data.get("arguments", {}),
                        "raw": match.group(0)
                    })
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse function code block: {match.group(1)}")

        return function_calls

    @staticmethod
    def validate_function_call(call: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        func_name = call.get("name")
        arguments = call.get("arguments", {})

        func_def = get_function_by_name(func_name)
        if not func_def:
            return False, f"Unknown function: {func_name}"

        params = func_def.get("parameters", {})
        required = params.get("required", [])
        
        for param in required:
            if param not in arguments:
                return False, f"Missing required parameter: {param}"

        properties = params.get("properties", {})
        for param_name, param_value in arguments.items():
            if param_name in properties:
                prop_def = properties[param_name]
                expected_type = prop_def.get("type")
                
                if expected_type == "string" and not isinstance(param_value, str):
                    return False, f"Parameter {param_name} must be a string"
                elif expected_type == "number" and not isinstance(param_value, (int, float)):
                    return False, f"Parameter {param_name} must be a number"
                elif expected_type == "integer" and not isinstance(param_value, int):
                    return False, f"Parameter {param_name} must be an integer"
                
                if "enum" in prop_def and param_value not in prop_def["enum"]:
                    return False, f"Parameter {param_name} must be one of: {prop_def['enum']}"

        return True, None

    @staticmethod
    def remove_function_calls(text: str) -> str:
        result = text

        patterns = [
            r'<function_call>\s*\{.*?\}\s*</function_call>',
            r'\[FUNCTION_CALL:\s*\w+\s*\(.*?\)\]',
            r'```function\s*\n\{.*?\}\s*\n```',
        ]

        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.DOTALL)

        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        return result.strip()

    @staticmethod
    def has_function_calls(text: str) -> bool:
        return bool(FunctionHandler.parse_function_calls(text))

    @staticmethod
    def format_function_result(name: str, result: Dict[str, Any]) -> str:
        if result.get("success"):
            if "memories" in result:
                memories = result["memories"]
                if memories:
                    mem_text = "\n".join([
                        f"- [{m.get('memory_type', 'fact')}] {m.get('content', '')}"
                        for m in memories[:5]
                    ])
                    return f"[Function {name} returned {len(memories)} memories:\n{mem_text}]"
                return f"[Function {name}: No memories found]"
            elif "profile" in result:
                profile = result["profile"]
                return f"[Function {name} returned profile: {json.dumps(profile, ensure_ascii=False)}]"
            else:
                return f"[Function {name}: {result.get('message', 'Success')}]"
        else:
            return f"[Function {name} failed: {result.get('error', 'Unknown error')}]"

