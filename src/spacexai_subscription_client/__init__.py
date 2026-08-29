"""Unofficial async client for SpaceXAI OAuth subscription APIs."""

from .client import SpaceXAISubscriptionClient
from .errors import (
    AuthenticationError,
    AuthorizationDeniedError,
    ConnectionFailureError,
    DeviceAuthorizationExpiredError,
    InvalidResponseError,
    RateLimitError,
    RequestTimeoutError,
    SpaceXAISubscriptionError,
)
from .models import (
    Account,
    Attachment,
    BuiltinTool,
    Completion,
    DeviceAuthorization,
    InputItem,
    Message,
    OAuthToken,
    ResponseTool,
    Tool,
    ToolCall,
    ToolResult,
)

__all__ = [
    "Account",
    "Attachment",
    "AuthenticationError",
    "AuthorizationDeniedError",
    "BuiltinTool",
    "Completion",
    "ConnectionFailureError",
    "DeviceAuthorization",
    "DeviceAuthorizationExpiredError",
    "InputItem",
    "InvalidResponseError",
    "Message",
    "OAuthToken",
    "RateLimitError",
    "RequestTimeoutError",
    "ResponseTool",
    "SpaceXAISubscriptionClient",
    "SpaceXAISubscriptionError",
    "Tool",
    "ToolCall",
    "ToolResult",
]
