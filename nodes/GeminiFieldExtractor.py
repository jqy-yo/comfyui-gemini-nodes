import json
import re
from typing import Any, Union, List


class GeminiFieldExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"multiline": True}),
                "field_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., name, user.email, items[0].price, data.users[*].name"
                }),
                "default_value": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Default value if field not found"
                }),
                "output_format": (["auto", "string", "json", "number", "boolean", "list"], {"default": "auto"}),
            },
            "optional": {
                "array_handling": (["first", "last", "all", "join"], {"default": "all"}),
                "join_separator": ("STRING", {"default": ", "}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("extracted_value", "formatted_output", "extraction_success")
    FUNCTION = "extract_field"
    CATEGORY = "🤖 Gemini"

    def __init__(self):
        self.last_error = None

    def _parse_json_safely(self, json_str: str) -> Union[dict, list, None]:
        try:
            json_str = json_str.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.startswith('```'):
                json_str = json_str[3:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            
            return json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            print(f"[GeminiFieldExtractor] JSON parse error: {str(e)}")
            self.last_error = f"Invalid JSON: {str(e)}"
            return None

    def _extract_by_path(self, data: Any, path: str) -> Any:
        if not path:
            return data
        
        path_parts = self._parse_path(path)
        current = data
        
        for part in path_parts:
            if part == '*':
                if isinstance(current, list):
                    results = []
                    for item in current:
                        if isinstance(item, dict):
                            results.append(item)
                    return results
                else:
                    return None
            
            elif isinstance(part, int):
                if isinstance(current, list):
                    if -len(current) <= part < len(current):
                        current = current[part]
                    else:
                        return None
                else:
                    return None
            
            elif isinstance(part, str):
                if part.endswith('[*]'):
                    key = part[:-3]
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                        if isinstance(current, list):
                            continue
                    return None
                
                elif '[' in part and ']' in part:
                    match = re.match(r'(\w+)\[(\d+)\]', part)
                    if match:
                        key, index = match.groups()
                        index = int(index)
                        if isinstance(current, dict) and key in current:
                            if isinstance(current[key], list):
                                if -len(current[key]) <= index < len(current[key]):
                                    current = current[key][index]
                                else:
                                    return None
                            else:
                                return None
                        else:
                            return None
                    else:
                        return None
                
                else:
                    if isinstance(current, dict):
                        if part in current:
                            current = current[part]
                        else:
                            return None
                    elif isinstance(current, list):
                        results = []
                        for item in current:
                            if isinstance(item, dict) and part in item:
                                results.append(item[part])
                        if results:
                            current = results
                        else:
                            return None
                    else:
                        return None
            else:
                return None
        
        return current

    def _parse_path(self, path: str) -> List[Union[str, int]]:
        parts = []
        current = ""
        
        for char in path:
            if char == '.':
                if current:
                    if current.isdigit():
                        parts.append(int(current))
                    else:
                        parts.append(current)
                    current = ""
            elif char == '[':
                if current:
                    parts.append(current)
                    current = ""
                bracket_content = ""
                continue_parse = True
            else:
                current += char
        
        if current:
            if current.isdigit():
                parts.append(int(current))
            else:
                parts.append(current)
        
        result = []
        temp_path = path
        while temp_path:
            if '.' in temp_path:
                dot_pos = temp_path.index('.')
                if '[' in temp_path[:dot_pos]:
                    bracket_start = temp_path.index('[')
                    bracket_end = temp_path.index(']')
                    key = temp_path[:bracket_start]
                    index = temp_path[bracket_start+1:bracket_end]
                    
                    if index == '*':
                        result.append(key + '[*]')
                    else:
                        result.append(f"{key}[{index}]")
                    
                    temp_path = temp_path[bracket_end+1:]
                    if temp_path.startswith('.'):
                        temp_path = temp_path[1:]
                else:
                    result.append(temp_path[:dot_pos])
                    temp_path = temp_path[dot_pos+1:]
            else:
                if '[' in temp_path:
                    bracket_start = temp_path.index('[')
                    bracket_end = temp_path.index(']')
                    key = temp_path[:bracket_start]
                    index = temp_path[bracket_start+1:bracket_end]
                    
                    if index == '*':
                        result.append(key + '[*]')
                    else:
                        result.append(f"{key}[{index}]")
                    break
                else:
                    result.append(temp_path)
                    break
        
        return result if result else parts

    def _format_value(self, value: Any, format_type: str, array_handling: str, separator: str) -> str:
        if value is None:
            return ""
        
        if isinstance(value, list) and array_handling != "all":
            if array_handling == "first" and value:
                value = value[0]
            elif array_handling == "last" and value:
                value = value[-1]
            elif array_handling == "join":
                value = separator.join(str(v) for v in value)
        
        if format_type == "auto":
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False, indent=2)
            else:
                return str(value)
        elif format_type == "string":
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)
        elif format_type == "json":
            return json.dumps(value, ensure_ascii=False, indent=2)
        elif format_type == "number":
            if isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, str):
                try:
                    return str(float(value))
                except:
                    return "0"
            else:
                return "0"
        elif format_type == "boolean":
            return str(bool(value)).lower()
        elif format_type == "list":
            if isinstance(value, list):
                return json.dumps(value, ensure_ascii=False)
            else:
                return json.dumps([value], ensure_ascii=False)
        
        return str(value)

    def extract_field(self, json_input: str, field_path: str, default_value: str, 
                     output_format: str, array_handling: str = "all", 
                     join_separator: str = ", "):
        
        self.last_error = None
        
        # Handle empty input
        if not json_input or json_input.strip() == "":
            print(f"[GeminiFieldExtractor] Empty input received - likely due to API error")
            return (default_value, "No input data (API might have failed)", False)
        
        # Check if input is an error message
        if json_input.startswith("Error:"):
            print(f"[GeminiFieldExtractor] Received error message instead of JSON: {json_input}")
            return (default_value, json_input, False)
        
        data = self._parse_json_safely(json_input)
        if data is None:
            print(f"[GeminiFieldExtractor] Failed to parse JSON input")
            # If it looks like a Gemini API error, provide more context
            if "500 INTERNAL" in json_input or "Internal error" in json_input:
                return (default_value, "API returned internal server error - model might not be available in your region", False)
            return (default_value, f"Error: {self.last_error}", False)
        
        try:
            extracted = self._extract_by_path(data, field_path)
            
            if extracted is None:
                print(f"[GeminiFieldExtractor] Field path '{field_path}' not found")
                return (default_value, f"Field not found: {field_path}", False)
            
            formatted = self._format_value(extracted, output_format, array_handling, join_separator)
            
            readable_output = self._create_readable_output(field_path, extracted, array_handling)
            
            print(f"[GeminiFieldExtractor] Successfully extracted field: {field_path}")
            return (formatted, readable_output, True)
            
        except Exception as e:
            error_msg = f"Extraction error: {str(e)}"
            print(f"[GeminiFieldExtractor] {error_msg}")
            return (default_value, error_msg, False)

    def _create_readable_output(self, path: str, value: Any, array_handling: str) -> str:
        lines = []
        lines.append(f"📍 Field Path: {path}")
        
        if isinstance(value, list):
            lines.append(f"📊 Type: Array ({len(value)} items)")
            if array_handling != "all":
                lines.append(f"🔧 Array Handling: {array_handling}")
        elif isinstance(value, dict):
            lines.append(f"📊 Type: Object ({len(value)} properties)")
        else:
            type_name = type(value).__name__
            lines.append(f"📊 Type: {type_name}")
        
        lines.append("📤 Extracted Value:")
        
        if isinstance(value, (dict, list)):
            lines.append(json.dumps(value, ensure_ascii=False, indent=2))
        else:
            lines.append(str(value))
        
        return "\n".join(lines)


class GeminiJSONParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"multiline": True}),
                "operation": (["validate", "format", "minify", "extract_keys", "get_type", "count_items"], 
                             {"default": "validate"}),
            },
            "optional": {
                "indent": ("INT", {"default": 2, "min": 0, "max": 8}),
                "sort_keys": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("result", "info", "success")
    FUNCTION = "parse_json"
    CATEGORY = "🤖 Gemini"

    def parse_json(self, json_input: str, operation: str, indent: int = 2, sort_keys: bool = False):
        
        json_input = json_input.strip()
        if json_input.startswith('```json'):
            json_input = json_input[7:]
        if json_input.startswith('```'):
            json_input = json_input[3:]
        if json_input.endswith('```'):
            json_input = json_input[:-3]
        
        try:
            data = json.loads(json_input.strip())
            
            if operation == "validate":
                result = "✅ Valid JSON"
                info = self._get_json_info(data)
                return (json.dumps(data, ensure_ascii=False, indent=2), info, True)
            
            elif operation == "format":
                result = json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
                info = f"Formatted with indent={indent}, sort_keys={sort_keys}"
                return (result, info, True)
            
            elif operation == "minify":
                result = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
                original_size = len(json_input)
                minified_size = len(result)
                info = f"Minified: {original_size} → {minified_size} bytes ({round(100 * minified_size / original_size, 1)}%)"
                return (result, info, True)
            
            elif operation == "extract_keys":
                keys = self._extract_all_keys(data)
                result = json.dumps(keys, ensure_ascii=False, indent=2)
                info = f"Found {len(keys)} unique keys"
                return (result, info, True)
            
            elif operation == "get_type":
                type_info = self._get_type_structure(data)
                result = json.dumps(type_info, ensure_ascii=False, indent=2)
                info = "Type structure analyzed"
                return (result, info, True)
            
            elif operation == "count_items":
                count_info = self._count_items(data)
                result = json.dumps(count_info, ensure_ascii=False, indent=2)
                info = f"Total items counted"
                return (result, info, True)
            
        except json.JSONDecodeError as e:
            error_msg = f"❌ Invalid JSON: {str(e)}"
            line_num = e.lineno if hasattr(e, 'lineno') else 'unknown'
            col_num = e.colno if hasattr(e, 'colno') else 'unknown'
            info = f"Error at line {line_num}, column {col_num}"
            return ("", f"{error_msg}\n{info}", False)
        except Exception as e:
            return ("", f"Error: {str(e)}", False)

    def _get_json_info(self, data: Any) -> str:
        info = []
        if isinstance(data, dict):
            info.append(f"Type: Object")
            info.append(f"Properties: {len(data)}")
            info.append(f"Keys: {', '.join(list(data.keys())[:5])}")
            if len(data) > 5:
                info.append(f"... and {len(data) - 5} more")
        elif isinstance(data, list):
            info.append(f"Type: Array")
            info.append(f"Length: {len(data)}")
            if data:
                first_type = type(data[0]).__name__
                info.append(f"First item type: {first_type}")
        else:
            info.append(f"Type: {type(data).__name__}")
            info.append(f"Value: {str(data)[:100]}")
        
        return "\n".join(info)

    def _extract_all_keys(self, data: Any, prefix: str = "") -> List[str]:
        keys = []
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.append(full_key)
                keys.extend(self._extract_all_keys(value, full_key))
        elif isinstance(data, list) and data:
            if isinstance(data[0], dict):
                keys.extend(self._extract_all_keys(data[0], f"{prefix}[0]" if prefix else "[0]"))
        return keys

    def _get_type_structure(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {key: self._get_type_structure(value) for key, value in data.items()}
        elif isinstance(data, list):
            if data:
                return [self._get_type_structure(data[0])]
            else:
                return []
        else:
            return type(data).__name__

    def _count_items(self, data: Any) -> dict:
        counts = {
            "total_keys": 0,
            "total_values": 0,
            "objects": 0,
            "arrays": 0,
            "strings": 0,
            "numbers": 0,
            "booleans": 0,
            "nulls": 0
        }
        
        def count_recursive(obj):
            if isinstance(obj, dict):
                counts["objects"] += 1
                counts["total_keys"] += len(obj)
                for value in obj.values():
                    count_recursive(value)
            elif isinstance(obj, list):
                counts["arrays"] += 1
                counts["total_values"] += len(obj)
                for item in obj:
                    count_recursive(item)
            elif isinstance(obj, str):
                counts["strings"] += 1
            elif isinstance(obj, (int, float)):
                counts["numbers"] += 1
            elif isinstance(obj, bool):
                counts["booleans"] += 1
            elif obj is None:
                counts["nulls"] += 1
        
        count_recursive(data)
        return counts