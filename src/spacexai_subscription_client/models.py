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
class Attachment:
    """Binary content attached to a user message."""

    filename: str
    media_type: str
    data: bytes


@dataclass(frozen=True, slots=True)
class Message:
    """Conversation message sent to Grok."""

    role: Literal["user", "assistant", "developer"]
    content: str
    attachments: tuple[Attachment, ...] = ()

    def __post_init__(self) -> None:
        """Validate attachment placement."""
        if self.attachments and self.role != "user":
            message = "Attachments are only valid on user messages"
            raise ValueError(message)


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


@dataclass(frozen=True, slots=True)
class BuiltinTool:
    """Provider-hosted tool enabled for a response."""

    type: Literal["web_search", "x_search", "code_interpreter"]


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    """Named JSON schema requested for a structured response."""

    name: str
    schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Prevent mutation of the JSON schema."""
        object.__setattr__(self, "schema", MappingProxyType(dict(self.schema)))


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """Image returned by the SpaceXAI Imagine API."""

    data: bytes
    media_type: str
    model: str
    revised_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class GeneratedVideo:
    """Video returned by the SpaceXAI Imagine API."""

    url: str
    model: str
    duration: int
    respect_moderation: bool


type InputItem = Message | ToolCall | ToolResult
type ResponseTool = Tool | BuiltinTool


@dataclass(frozen=True, slots=True)
class Completion:
    """Normalized Grok completion."""

    text: str
    tool_calls: tuple[ToolCall, ...]
