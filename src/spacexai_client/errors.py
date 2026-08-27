"""Exceptions raised by the SpaceXAI client."""


class SpaceXAIError(Exception):
    """Base class for SpaceXAI client failures."""


class AuthenticationError(SpaceXAIError):
    """SpaceXAI rejected the current access token or OAuth client."""


class AuthorizationDeniedError(AuthenticationError):
    """The user denied device authorization."""


class ConnectionFailureError(SpaceXAIError):
    """SpaceXAI could not be reached."""


class InvalidResponseError(SpaceXAIError):
    """SpaceXAI returned a response that did not match its contract."""


class RateLimitError(SpaceXAIError):
    """SpaceXAI rate limited the request."""


class RequestTimeoutError(SpaceXAIError):
    """A SpaceXAI request or device authorization timed out."""
