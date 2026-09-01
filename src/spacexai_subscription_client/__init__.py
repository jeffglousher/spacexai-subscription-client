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
    Completion,
    DeviceAuthorization,
    InputItem,
    Message,
    OAuthToken,
    Tool,
    ToolCall,
    ToolResult,
)

__all__ = [
    "Account",
    "AuthenticationError",
    "AuthorizationDeniedError",
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
    "SpaceXAISubscriptionClient",
    "SpaceXAISubscriptionError",
    "Tool",
    "ToolCall",
    "ToolResult",
]
