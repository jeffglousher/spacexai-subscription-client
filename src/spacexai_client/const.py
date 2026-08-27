"""Constants for the SpaceXAI client."""

AUTHORIZE_URL = "https://auth.x.ai/oauth2/authorize"
TOKEN_URL = "https://auth.x.ai/oauth2/token"  # noqa: S105
USERINFO_URL = "https://auth.x.ai/oauth2/userinfo"
DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
API_BASE_URL = "https://cli-chat-proxy.grok.com/v1"
GROK_CLI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
GROK_CLI_REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "spacexai-client/0.1.0",
    "x-xai-token-auth": "xai-grok-cli",
    "x-grok-client-identifier": "spacexai-client",
    "x-grok-client-version": "0.1.0",
}
OAUTH_SCOPES = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "grok-cli:access",
)

DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
DEVICE_CODE_MAX_POLL_SECONDS = 900
HTTP_TIMEOUT = 30
