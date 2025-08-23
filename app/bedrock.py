import json
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union
import re

import boto3


# Global variables to track the current tool use ID across function calls
# Tmp solution
CURRENT_TOOLUSE_ID = None


# Model Type Enumeration
class ModelType(Enum):
    """Enumeration of supported model types"""
    CLAUDE = "claude"        # Anthropic Claude models
    NOVA_PRO = "nova_pro"    # Amazon Nova Pro models
    UNKNOWN = "unknown"      # Unsupported or unrecognized models


# Model Configuration
@dataclass
class ModelConfig:
    """Configuration settings for a specific model"""
    model_id: str              # Full Bedrock model ID (e.g., us.anthropic.claude-3-7-sonnet-20250219-v1:0)
    model_type: ModelType      # Type of model (Claude, Nova Pro, etc.)
    family: str               # Model family (e.g., claude-3-7-sonnet, nova-pro)
    supports_streaming: bool   # Whether the model supports streaming responses
    supports_tools: bool      # Whether the model supports function calling
    max_tokens: int           # Maximum tokens per request
    default_temperature: float # Recommended temperature setting for this model


# Model Error Exception
class ModelError(Exception):
    """Exception raised for model-specific errors
    
    Attributes:
        message: Human-readable error message
        model_id: The model ID that caused the error
        error_type: Category of error (INVALID_MODEL_ID, UNSUPPORTED_MODEL, etc.)
    """
    def __init__(self, message: str, model_id: str, error_type: str):
        self.message = message
        self.model_id = model_id
        self.error_type = error_type
        super().__init__(f"[{model_id}] {error_type}: {message}")


# Model Detection Service
class ModelDetector:
    """Service for detecting and validating model types from model IDs
    
    Supports:
    - Claude models: *.anthropic.claude-* (e.g., us.anthropic.claude-3-7-sonnet-20250219-v1:0)
    - Nova Pro models: *.amazon.nova-* (e.g., us.amazon.nova-pro-v1:0)
    """
    # Model ID patterns for automatic detection
    CLAUDE_PATTERN = re.compile(r'.*\.anthropic\.claude-.*')
    NOVA_PRO_PATTERN = re.compile(r'.*\.amazon\.nova-.*')
    
    @staticmethod
    def get_model_type(model_id: str) -> ModelType:
        """Detect model type from model ID"""
        if not model_id:
            return ModelType.UNKNOWN
            
        if ModelDetector.CLAUDE_PATTERN.match(model_id):
            return ModelType.CLAUDE
        elif ModelDetector.NOVA_PRO_PATTERN.match(model_id):
            return ModelType.NOVA_PRO
        else:
            return ModelType.UNKNOWN
    
    @staticmethod
    def validate_model_id(model_id: str) -> bool:
        """Validate model ID format"""
        if not model_id or not isinstance(model_id, str):
            return False
        
        model_type = ModelDetector.get_model_type(model_id)
        return model_type != ModelType.UNKNOWN
    
    @staticmethod
    def get_model_family(model_id: str) -> str:
        """Get model family from model ID"""
        model_type = ModelDetector.get_model_type(model_id)
        
        if model_type == ModelType.CLAUDE:
            # Extract Claude model family (e.g., "claude-3-7-sonnet")
            match = re.search(r'claude-([^:]+)', model_id)
            return match.group(1) if match else "claude"
        elif model_type == ModelType.NOVA_PRO:
            # Extract Nova model family (e.g., "nova-pro")
            match = re.search(r'nova-([^:]+)', model_id)
            return match.group(1) if match else "nova-pro"
        else:
            return "unknown"
    
    @staticmethod
    def get_model_config(model_id: str) -> ModelConfig:
        """Get model configuration based on model ID"""
        model_type = ModelDetector.get_model_type(model_id)
        family = ModelDetector.get_model_family(model_id)
        
        if model_type == ModelType.CLAUDE:
            return ModelConfig(
                model_id=model_id,
                model_type=model_type,
                family=family,
                supports_streaming=True,
                supports_tools=True,
                max_tokens=8192,
                default_temperature=1.0
            )
        elif model_type == ModelType.NOVA_PRO:
            return ModelConfig(
                model_id=model_id,
                model_type=model_type,
                family=family,
                supports_streaming=True,
                supports_tools=True,
                max_tokens=8192,
                default_temperature=0.7
            )
        else:
            raise ModelError(
                f"Unsupported model ID: {model_id}. Supported patterns: *.anthropic.claude-* or *.amazon.nova-*",
                model_id,
                "UNSUPPORTED_MODEL"
            )


# Base Model Handler Interface
class BaseModelHandler(ABC):
    """Abstract base class for model-specific operations"""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
    
    @abstractmethod
    def format_messages(self, messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Format messages for the specific model"""
        pass
    
    @abstractmethod
    def format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Format tools for the specific model"""
        pass
    
    @abstractmethod
    def format_response(self, response: Dict) -> 'OpenAIResponse':
        """Format response from the specific model"""
        pass
    
    @abstractmethod
    def get_api_parameters(self, **kwargs) -> Dict:
        """Get API parameters for the specific model"""
        pass


# Model Handler Factory
class ModelHandlerFactory:
    """Factory for creating appropriate model handlers"""
    
    @staticmethod
    def create_handler(model_id: str) -> BaseModelHandler:
        """Create appropriate handler based on model ID"""
        model_config = ModelDetector.get_model_config(model_id)
        
        if model_config.model_type == ModelType.CLAUDE:
            return ClaudeHandler(model_config)
        elif model_config.model_type == ModelType.NOVA_PRO:
            return NovaProHandler(model_config)
        else:
            raise ModelError(
                f"No handler available for model type: {model_config.model_type}",
                model_id,
                "NO_HANDLER"
            )


# Claude Handler Implementation
class ClaudeHandler(BaseModelHandler):
    """Handler for Claude models with existing functionality"""
    
    def format_messages(self, messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Format messages for Claude model"""
        bedrock_messages = []
        system_prompt = []
        for message in messages:
            if message.get("role") == "system":
                system_prompt = [{"text": message.get("content")}]
            elif message.get("role") == "user":
                bedrock_message = {
                    "role": message.get("role", "user"),
                    "content": [{"text": message.get("content")}],
                }
                bedrock_messages.append(bedrock_message)
            elif message.get("role") == "assistant":
                bedrock_message = {
                    "role": "assistant",
                    "content": [{"text": message.get("content")}],
                }
                openai_tool_calls = message.get("tool_calls", [])
                if openai_tool_calls:
                    bedrock_tool_use = {
                        "toolUseId": openai_tool_calls[0]["id"],
                        "name": openai_tool_calls[0]["function"]["name"],
                        "input": json.loads(
                            openai_tool_calls[0]["function"]["arguments"]
                        ),
                    }
                    bedrock_message["content"].append({"toolUse": bedrock_tool_use})
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = openai_tool_calls[0]["id"]
                bedrock_messages.append(bedrock_message)
            elif message.get("role") == "tool":
                bedrock_message = {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": CURRENT_TOOLUSE_ID,
                                "content": [{"text": message.get("content")}],
                            }
                        }
                    ],
                }
                bedrock_messages.append(bedrock_message)
            else:
                raise ValueError(f"Invalid role: {message.get('role')}")
        return system_prompt, bedrock_messages
    
    def format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Format tools for Claude model"""
        bedrock_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                bedrock_tool = {
                    "toolSpec": {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": function.get("parameters", {}).get(
                                    "properties", {}
                                ),
                                "required": function.get("parameters", {}).get(
                                    "required", []
                                ),
                            }
                        },
                    }
                }
                bedrock_tools.append(bedrock_tool)
        return bedrock_tools
    
    def format_response(self, bedrock_response: Dict) -> OpenAIResponse:
        """Format Claude response to OpenAI format"""
        content = ""
        if bedrock_response.get("output", {}).get("message", {}).get("content"):
            content_array = bedrock_response["output"]["message"]["content"]
            content = "".join(item.get("text", "") for item in content_array)
        if content == "":
            content = "."

        # Handle tool calls in response
        openai_tool_calls = []
        if bedrock_response.get("output", {}).get("message", {}).get("content"):
            for content_item in bedrock_response["output"]["message"]["content"]:
                if content_item.get("toolUse"):
                    bedrock_tool_use = content_item["toolUse"]
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = bedrock_tool_use["toolUseId"]
                    openai_tool_call = {
                        "id": CURRENT_TOOLUSE_ID,
                        "type": "function",
                        "function": {
                            "name": bedrock_tool_use["name"],
                            "arguments": json.dumps(bedrock_tool_use["input"]),
                        },
                    }
                    openai_tool_calls.append(openai_tool_call)

        # Construct final OpenAI format response
        openai_format = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "created": int(time.time()),
            "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [
                {
                    "finish_reason": bedrock_response.get("stopReason", "end_turn"),
                    "index": 0,
                    "message": {
                        "content": content,
                        "role": bedrock_response.get("output", {})
                        .get("message", {})
                        .get("role", "assistant"),
                        "tool_calls": openai_tool_calls
                        if openai_tool_calls != []
                        else None,
                        "function_call": None,
                    },
                }
            ],
            "usage": {
                "completion_tokens": bedrock_response.get("usage", {}).get(
                    "outputTokens", 0
                ),
                "prompt_tokens": bedrock_response.get("usage", {}).get(
                    "inputTokens", 0
                ),
                "total_tokens": bedrock_response.get("usage", {}).get("totalTokens", 0),
            },
        }
        return OpenAIResponse(openai_format)
    
    def get_api_parameters(self, **kwargs) -> Dict:
        """Get API parameters for Claude model"""
        return {
            "modelId": self.model_config.model_id,
            "inferenceConfig": {
                "temperature": kwargs.get("temperature", self.model_config.default_temperature),
                "maxTokens": kwargs.get("max_tokens", self.model_config.max_tokens)
            }
        }


# Nova Pro Handler Implementation
class NovaProHandler(BaseModelHandler):
    """Handler for Amazon Nova Pro models"""
    
    def format_messages(self, messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Format messages for Nova Pro model"""
        # Nova Pro uses similar message format to Claude but may have different requirements
        bedrock_messages = []
        system_prompt = []
        
        for message in messages:
            if message.get("role") == "system":
                system_prompt = [{"text": message.get("content")}]
            elif message.get("role") == "user":
                bedrock_message = {
                    "role": message.get("role", "user"),
                    "content": [{"text": message.get("content")}],
                }
                bedrock_messages.append(bedrock_message)
            elif message.get("role") == "assistant":
                bedrock_message = {
                    "role": "assistant",
                    "content": [{"text": message.get("content")}],
                }
                openai_tool_calls = message.get("tool_calls", [])
                if openai_tool_calls:
                    bedrock_tool_use = {
                        "toolUseId": openai_tool_calls[0]["id"],
                        "name": openai_tool_calls[0]["function"]["name"],
                        "input": json.loads(
                            openai_tool_calls[0]["function"]["arguments"]
                        ),
                    }
                    bedrock_message["content"].append({"toolUse": bedrock_tool_use})
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = openai_tool_calls[0]["id"]
                bedrock_messages.append(bedrock_message)
            elif message.get("role") == "tool":
                bedrock_message = {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": CURRENT_TOOLUSE_ID,
                                "content": [{"text": message.get("content")}],
                            }
                        }
                    ],
                }
                bedrock_messages.append(bedrock_message)
            else:
                raise ValueError(f"Invalid role: {message.get('role')}")
        return system_prompt, bedrock_messages
    
    def format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Format tools for Nova Pro model"""
        # Nova Pro uses the same tool format as Claude for now
        bedrock_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                bedrock_tool = {
                    "toolSpec": {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": function.get("parameters", {}).get(
                                    "properties", {}
                                ),
                                "required": function.get("parameters", {}).get(
                                    "required", []
                                ),
                            }
                        },
                    }
                }
                bedrock_tools.append(bedrock_tool)
        return bedrock_tools
    
    def format_response(self, bedrock_response: Dict) -> OpenAIResponse:
        """Format Nova Pro response to OpenAI format"""
        content = ""
        if bedrock_response.get("output", {}).get("message", {}).get("content"):
            content_array = bedrock_response["output"]["message"]["content"]
            content = "".join(item.get("text", "") for item in content_array)
        if content == "":
            content = "."

        # Handle tool calls in response
        openai_tool_calls = []
        if bedrock_response.get("output", {}).get("message", {}).get("content"):
            for content_item in bedrock_response["output"]["message"]["content"]:
                if content_item.get("toolUse"):
                    bedrock_tool_use = content_item["toolUse"]
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = bedrock_tool_use["toolUseId"]
                    openai_tool_call = {
                        "id": CURRENT_TOOLUSE_ID,
                        "type": "function",
                        "function": {
                            "name": bedrock_tool_use["name"],
                            "arguments": json.dumps(bedrock_tool_use["input"]),
                        },
                    }
                    openai_tool_calls.append(openai_tool_call)

        # Construct final OpenAI format response
        openai_format = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "created": int(time.time()),
            "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [
                {
                    "finish_reason": bedrock_response.get("stopReason", "end_turn"),
                    "index": 0,
                    "message": {
                        "content": content,
                        "role": bedrock_response.get("output", {})
                        .get("message", {})
                        .get("role", "assistant"),
                        "tool_calls": openai_tool_calls
                        if openai_tool_calls != []
                        else None,
                        "function_call": None,
                    },
                }
            ],
            "usage": {
                "completion_tokens": bedrock_response.get("usage", {}).get(
                    "outputTokens", 0
                ),
                "prompt_tokens": bedrock_response.get("usage", {}).get(
                    "inputTokens", 0
                ),
                "total_tokens": bedrock_response.get("usage", {}).get("totalTokens", 0),
            },
        }
        return OpenAIResponse(openai_format)
    
    def get_api_parameters(self, **kwargs) -> Dict:
        """Get API parameters for Nova Pro model"""
        return {
            "modelId": self.model_config.model_id,
            "inferenceConfig": {
                "temperature": kwargs.get("temperature", self.model_config.default_temperature),
                "maxTokens": kwargs.get("max_tokens", self.model_config.max_tokens)
            }
        }


# Class to handle OpenAI-style response formatting
class OpenAIResponse:
    def __init__(self, data):
        # Recursively convert nested dicts and lists to OpenAIResponse objects
        for key, value in data.items():
            if isinstance(value, dict):
                value = OpenAIResponse(value)
            elif isinstance(value, list):
                value = [
                    OpenAIResponse(item) if isinstance(item, dict) else item
                    for item in value
                ]
            setattr(self, key, value)

    def model_dump(self, *args, **kwargs):
        # Convert object to dict and add timestamp
        data = self.__dict__
        data["created_at"] = datetime.now().isoformat()
        return data


# Main client class for interacting with Amazon Bedrock
class BedrockClient:
    def __init__(self, model_id: Optional[str] = None):
        # Initialize Bedrock client, you need to configure AWS env first
        try:
            self.client = boto3.client("bedrock-runtime")
            self.chat = Chat(self.client)
            
            # Validate model if provided
            if model_id:
                self._validate_model_on_init(model_id)
                
        except Exception as e:
            print(f"Error initializing Bedrock client: {e}")
            sys.exit(1)
    
    def _validate_model_on_init(self, model_id: str):
        """Validate model during initialization"""
        try:
            if not ModelDetector.validate_model_id(model_id):
                print(f"Warning: Model ID '{model_id}' does not match expected patterns.")
                print("Supported patterns:")
                print("  - Claude: *.anthropic.claude-*")
                print("  - Nova Pro: *.amazon.nova-*")
                print("Falling back to backward compatibility mode.")
            else:
                model_type = ModelDetector.get_model_type(model_id)
                print(f"Validated model: {model_id} (type: {model_type.value})")
        except Exception as e:
            print(f"Model validation warning: {e}")


# Chat interface class
class Chat:
    def __init__(self, client):
        self.completions = ChatCompletions(client)


# Core class handling chat completions functionality
class ChatCompletions:
    def __init__(self, client):
        self.client = client
        self._handler_cache = {}  # Cache handlers for performance

    def _get_model_handler(self, model_id: str) -> BaseModelHandler:
        """Get or create model handler for the given model ID"""
        if model_id not in self._handler_cache:
            try:
                # Validate model ID first
                if not ModelDetector.validate_model_id(model_id):
                    raise ModelError(
                        f"Invalid model ID format: {model_id}. Expected patterns: *.anthropic.claude-* or *.amazon.nova-*",
                        model_id,
                        "INVALID_FORMAT"
                    )
                
                self._handler_cache[model_id] = ModelHandlerFactory.create_handler(model_id)
                print(f"Created handler for model: {model_id} (type: {ModelDetector.get_model_type(model_id).value})")
                
            except ModelError as e:
                print(f"Model error for {model_id}: {e}")
                # Check if it's a legacy Claude model ID (backward compatibility)
                if "claude" in model_id.lower():
                    print(f"Attempting backward compatibility for Claude model: {model_id}")
                    model_config = ModelConfig(
                        model_id=model_id,
                        model_type=ModelType.CLAUDE,
                        family="claude",
                        supports_streaming=True,
                        supports_tools=True,
                        max_tokens=8192,
                        default_temperature=1.0
                    )
                    self._handler_cache[model_id] = ClaudeHandler(model_config)
                else:
                    # Re-raise the error for non-Claude models
                    raise e
            except Exception as e:
                raise ModelError(
                    f"Unexpected error creating handler: {str(e)}",
                    model_id,
                    "HANDLER_CREATION_ERROR"
                )
        return self._handler_cache[model_id]

    def _convert_openai_tools_to_bedrock_format(self, tools, model_id: str):
        # Use model handler for tool formatting
        handler = self._get_model_handler(model_id)
        return handler.format_tools(tools)

    def _convert_openai_messages_to_bedrock_format(self, messages, model_id: str):
        # Use model handler for message formatting
        handler = self._get_model_handler(model_id)
        return handler.format_messages(messages)

    def _convert_bedrock_response_to_openai_format(self, bedrock_response, model_id: str):
        # Use model handler for response formatting
        handler = self._get_model_handler(model_id)
        return handler.format_response(bedrock_response)

    async def _invoke_bedrock(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        # Non-streaming invocation of Bedrock model
        (
            system_prompt,
            bedrock_messages,
        ) = self._convert_openai_messages_to_bedrock_format(messages, model)
        
        # Get model handler for API parameters
        handler = self._get_model_handler(model)
        api_params = handler.get_api_parameters(
            temperature=temperature, 
            max_tokens=max_tokens
        )
        
        response = self.client.converse(
            modelId=model,
            system=system_prompt,
            messages=bedrock_messages,
            inferenceConfig=api_params["inferenceConfig"],
            toolConfig={"tools": tools} if tools else None,
        )
        openai_response = self._convert_bedrock_response_to_openai_format(response, model)
        return openai_response

    async def _invoke_bedrock_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        # Streaming invocation of Bedrock model
        (
            system_prompt,
            bedrock_messages,
        ) = self._convert_openai_messages_to_bedrock_format(messages, model)
        
        # Get model handler for API parameters
        handler = self._get_model_handler(model)
        api_params = handler.get_api_parameters(
            temperature=temperature, 
            max_tokens=max_tokens
        )
        
        response = self.client.converse_stream(
            modelId=model,
            system=system_prompt,
            messages=bedrock_messages,
            inferenceConfig=api_params["inferenceConfig"],
            toolConfig={"tools": tools} if tools else None,
        )

        # Initialize response structure
        bedrock_response = {
            "output": {"message": {"role": "", "content": []}},
            "stopReason": "",
            "usage": {},
            "metrics": {},
        }
        bedrock_response_text = ""
        bedrock_response_tool_input = ""

        # Process streaming response
        stream = response.get("stream")
        if stream:
            for event in stream:
                if event.get("messageStart", {}).get("role"):
                    bedrock_response["output"]["message"]["role"] = event[
                        "messageStart"
                    ]["role"]
                if event.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                    bedrock_response_text += event["contentBlockDelta"]["delta"]["text"]
                    print(
                        event["contentBlockDelta"]["delta"]["text"], end="", flush=True
                    )
                if event.get("contentBlockStop", {}).get("contentBlockIndex") == 0:
                    bedrock_response["output"]["message"]["content"].append(
                        {"text": bedrock_response_text}
                    )
                if event.get("contentBlockStart", {}).get("start", {}).get("toolUse"):
                    bedrock_tool_use = event["contentBlockStart"]["start"]["toolUse"]
                    tool_use = {
                        "toolUseId": bedrock_tool_use["toolUseId"],
                        "name": bedrock_tool_use["name"],
                    }
                    bedrock_response["output"]["message"]["content"].append(
                        {"toolUse": tool_use}
                    )
                    global CURRENT_TOOLUSE_ID
                    CURRENT_TOOLUSE_ID = bedrock_tool_use["toolUseId"]
                if event.get("contentBlockDelta", {}).get("delta", {}).get("toolUse"):
                    bedrock_response_tool_input += event["contentBlockDelta"]["delta"][
                        "toolUse"
                    ]["input"]
                    print(
                        event["contentBlockDelta"]["delta"]["toolUse"]["input"],
                        end="",
                        flush=True,
                    )
                if event.get("contentBlockStop", {}).get("contentBlockIndex") == 1:
                    bedrock_response["output"]["message"]["content"][1]["toolUse"][
                        "input"
                    ] = json.loads(bedrock_response_tool_input)
        print()
        openai_response = self._convert_bedrock_response_to_openai_format(
            bedrock_response, model
        )
        return openai_response

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: Optional[bool] = True,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        **kwargs,
    ) -> OpenAIResponse:
        # Main entry point for chat completion
        # Validate model ID
        if not ModelDetector.validate_model_id(model):
            raise ModelError(
                f"Invalid or unsupported model ID: {model}. Supported patterns: *.anthropic.claude-* or *.amazon.nova-*",
                model,
                "INVALID_MODEL_ID"
            )
        
        bedrock_tools = []
        if tools is not None:
            bedrock_tools = self._convert_openai_tools_to_bedrock_format(tools, model)
        if stream:
            return self._invoke_bedrock_stream(
                model,
                messages,
                max_tokens,
                temperature,
                bedrock_tools,
                tool_choice,
                **kwargs,
            )
        else:
            return self._invoke_bedrock(
                model,
                messages,
                max_tokens,
                temperature,
                bedrock_tools,
                tool_choice,
                **kwargs,
            )
