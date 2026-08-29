"""Constants for the unofficial SpaceXAI subscription client."""

from importlib.metadata import version

PACKAGE_VERSION = version("spacexai-subscription-client")

AUTHORIZE_URL = "https://auth.x.ai/oauth2/authorize"
TOKEN_URL = "https://auth.x.ai/oauth2/token"  # noqa: S105
USERINFO_URL = "https://auth.x.ai/oauth2/userinfo"
DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
API_BASE_URL = "https://cli-chat-proxy.grok.com/v1"
DEVELOPER_API_BASE_URL = "https://api.x.ai/v1"
IMAGES_EDIT_URL = f"{DEVELOPER_API_BASE_URL}/images/edits"
IMAGES_URL = f"{DEVELOPER_API_BASE_URL}/images/generations"
STT_URL = f"{DEVELOPER_API_BASE_URL}/stt"
TTS_URL = f"{DEVELOPER_API_BASE_URL}/tts"
GROK_CLI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
GROK_OAUTH_REQUEST_HEADERS = {
    "x-grok-client-surface": "ui",
    "x-grok-client-version": PACKAGE_VERSION,
}
GROK_CLI_REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": f"spacexai-subscription-client/{PACKAGE_VERSION}",
    "x-xai-token-auth": "xai-grok-cli",
    "x-grok-client-identifier": "spacexai-subscription-client",
    "x-grok-client-version": PACKAGE_VERSION,
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
IMAGE_TIMEOUT = 120
MAX_IMAGE_SIZE = 20 * 1024 * 1024
MAX_STT_SIZE = 500 * 1024 * 1024
MAX_TTS_SIZE = 50 * 1024 * 1024
MODEL_CATALOG_TIMEOUT = 10.0
RESPONSE_TIMEOUT = 120.0
SDK_MAX_RETRIES = 0
SPEECH_TIMEOUT = 120
TTS_MAX_TEXT_LENGTH = 15_000
TTS_MAX_SPEED = 1.5
TTS_MIN_SPEED = 0.7
