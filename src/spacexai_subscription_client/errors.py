"""Exceptions raised by the SpaceXAI subscription client."""


class SpaceXAISubscriptionError(Exception):
    """Base class for SpaceXAI subscription client failures."""


class AuthenticationError(SpaceXAISubscriptionError):
    """SpaceXAI rejected the current access token or OAuth client."""


class AuthorizationDeniedError(AuthenticationError):
    """The user denied device authorization."""


class ConnectionFailureError(SpaceXAISubscriptionError):
    """SpaceXAI could not be reached."""


class DeviceAuthorizationExpiredError(SpaceXAISubscriptionError):
    """The OAuth device authorization expired before it was approved."""


class InvalidResponseError(SpaceXAISubscriptionError):
    """SpaceXAI returned a response that did not match its contract."""


class RateLimitError(SpaceXAISubscriptionError):
    """SpaceXAI rate limited the request."""


class RequestTimeoutError(SpaceXAISubscriptionError):
    """A SpaceXAI request timed out."""
