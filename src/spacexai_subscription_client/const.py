"""Constants for the unofficial SpaceXAI subscription client."""

AUTHORIZE_URL = "https://auth.x.ai/oauth2/authorize"
TOKEN_URL = "https://auth.x.ai/oauth2/token"  # noqa: S105
USERINFO_URL = "https://auth.x.ai/oauth2/userinfo"
DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
API_BASE_URL = "https://cli-chat-proxy.grok.com/v1"
GROK_CLI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
GROK_OAUTH_REQUEST_HEADERS = {
    "x-grok-client-surface": "ui",
    "x-grok-client-version": "0.1.0",
}
GROK_CLI_REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "spacexai-subscription-client/0.1.0",
    "x-xai-token-auth": "xai-grok-cli",
    "x-grok-client-identifier": "spacexai-subscription-client",
    "x-grok-client-version": "0.1.0",
}
OAUTH_SCOPES = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "grok-cli:access",
    "api:access",
    "conversations:read",
    "conversations:write",
    "workspaces:read",
    "workspaces:write",
)
OAUTH_REFERRER = "grok-build"

DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
HTTP_TIMEOUT = 30
MODEL_CATALOG_TIMEOUT = 10.0
RESPONSE_TIMEOUT = 120.0
SDK_MAX_RETRIES = 0
