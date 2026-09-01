"""Data models exposed by the Grok subscription client."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class DeviceAuthorization:
    """Device authorization details shown to a user."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


@dataclass(frozen=True, slots=True)
class OAuthToken:
    """Validated OAuth token response."""

    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Prevent mutation of the provider token payload."""
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable token mapping for caller-owned persistence."""
        return dict(self.data)


@dataclass(frozen=True, slots=True)
class Account:
    """Authenticated Grok account identity."""

    subject: str
    name: str | None
    email: str | None

    @property
    def display_name(self) -> str:
        """Return the best available account name."""
        return self.name or self.email or "SpaceXAI"


@dataclass(frozen=True, slots=True)
class Message:
    """Conversation message sent to Grok."""

    role: Literal["user", "assistant", "developer"]
    content: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Client-side tool call."""

    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Prevent mutation of tool arguments."""
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Serialized result of a client-side tool call."""

    call_id: str
    output: str


@dataclass(frozen=True, slots=True)
class Tool:
    """Client-side tool definition."""

    name: str
    description: str | None
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Prevent mutation of tool parameters."""
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


type InputItem = Message | ToolCall | ToolResult


@dataclass(frozen=True, slots=True)
class Completion:
    """Normalized Grok completion."""

    text: str
    tool_calls: tuple[ToolCall, ...]
