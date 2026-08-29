"""Tests for the public Grok subscription client."""

from collections.abc import Generator
from typing import Any, Self
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import openai
import pytest
from aiohttp import ClientError
from httpx import Request, Response
from openai.types.responses import ResponseFunctionToolCall

from spacexai_subscription_client import (
    Account,
    Attachment,
    AuthenticationError,
    AuthorizationDeniedError,
    BuiltinTool,
    ConnectionFailureError,
    DeviceAuthorization,
    DeviceAuthorizationExpiredError,
    InvalidResponseError,
    Message,
    RateLimitError,
    RequestTimeoutError,
    SpaceXAISubscriptionClient,
    SpaceXAISubscriptionError,
    Tool,
    ToolCall,
    ToolResult,
)
from spacexai_subscription_client.const import (
    GROK_CLI_OAUTH_CLIENT_ID,
    GROK_CLI_REQUEST_HEADERS,
    GROK_OAUTH_REQUEST_HEADERS,
    MODEL_CATALOG_TIMEOUT,
    OAUTH_REFERRER,
    OAUTH_SCOPES,
    RESPONSE_TIMEOUT,
    SDK_MAX_RETRIES,
)


class MockResponse:
    """Minimal async response context."""

    def __init__(self, status: int, payload: object | Exception) -> None:
        """Initialize a response."""
        self.status = status
        self._payload = payload

    async def __aenter__(self) -> Self:
        """Enter the response context."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the response context."""

    async def json(self, *, content_type: None = None) -> object:
        """Return the configured JSON payload."""
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _authorization(*, expires_in: int = 1800) -> DeviceAuthorization:
    """Return device authorization details."""
    return DeviceAuthorization(
        "device-code",
        "ABCD-1234",
        "https://auth.x.ai/device",
        "https://auth.x.ai/device",
        expires_in,
        1,
    )


_REQUEST = Request("POST", "https://api.example.test")
_RESPONSE = Response(401, request=_REQUEST)
SDK_ERRORS = (
    pytest.param(
        openai.AuthenticationError("rejected", response=_RESPONSE, body=None),
        AuthenticationError,
        id="authentication",
    ),
    pytest.param(
        openai.APITimeoutError(_REQUEST),
        RequestTimeoutError,
        id="timeout",
    ),
    pytest.param(
        openai.APIConnectionError(request=_REQUEST),
        ConnectionFailureError,
        id="connection",
    ),
    pytest.param(
        openai.RateLimitError("limited", response=_RESPONSE, body=None),
        RateLimitError,
        id="rate_limit",
    ),
    pytest.param(openai.OpenAIError("failed"), SpaceXAISubscriptionError, id="sdk"),
)


@pytest.fixture
def websession() -> MagicMock:
    """Return a mocked aiohttp session."""
    return MagicMock()


@pytest.fixture
def client(websession: MagicMock) -> SpaceXAISubscriptionClient:
    """Return a Grok subscription client."""
    return SpaceXAISubscriptionClient(websession, MagicMock())


@pytest.fixture
def sdk() -> Generator[MagicMock]:
    """Mock the provider SDK client."""
    sdk = MagicMock()
    sdk.models.list = AsyncMock()
    sdk.responses.create = AsyncMock()
    with patch(
        "spacexai_subscription_client.client.openai.AsyncOpenAI", return_value=sdk
    ) as constructor:
        sdk.constructor = constructor
        yield sdk


async def test_device_authorization(
    client: SpaceXAISubscriptionClient, websession: MagicMock
) -> None:
    """Parse a successful device authorization response."""
    websession.post.return_value = MockResponse(
        200,
        {
            "device_code": "device-code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://auth.x.ai/device",
            "expires_in": 1800,
        },
    )

    authorization = await client.async_request_device_authorization()

    assert authorization.user_code == "ABCD-1234"
    assert authorization.verification_uri_complete == "https://auth.x.ai/device"
    assert authorization.interval == 5
    websession.post.assert_called_once_with(
        "https://auth.x.ai/oauth2/device/code",
        data={
            "client_id": GROK_CLI_OAUTH_CLIENT_ID,
            "referrer": OAUTH_REFERRER,
            "scope": " ".join(OAUTH_SCOPES),
        },
        headers=GROK_OAUTH_REQUEST_HEADERS,
        timeout=ANY,
    )


@pytest.mark.parametrize(
    ("side_effect", "expected_error"),
    [
        pytest.param(TimeoutError(), RequestTimeoutError, id="timeout"),
        pytest.param(ClientError(), ConnectionFailureError, id="connection"),
    ],
)
async def test_device_authorization_transport_error(
    client: SpaceXAISubscriptionClient,
    expected_error: type[SpaceXAISubscriptionError],
    side_effect: Exception,
    websession: MagicMock,
) -> None:
    """Translate transport failures while requesting a device code."""
    websession.post.side_effect = side_effect

    with pytest.raises(expected_error):
        await client.async_request_device_authorization()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing_fields"),
        pytest.param(
            {
                "device_code": "device-code",
                "user_code": "ABCD-1234",
                "verification_uri": "https://auth.x.ai/device",
                "verification_uri_complete": 123,
                "expires_in": 1800,
            },
            id="invalid_verification_uri",
        ),
        pytest.param(
            {
                "device_code": "device-code",
                "user_code": "ABCD-1234",
                "verification_uri": "https://auth.x.ai/device",
                "expires_in": 0,
            },
            id="invalid_expiry",
        ),
        pytest.param(ValueError(), id="invalid_json"),
    ],
)
async def test_device_authorization_invalid_response(
    client: SpaceXAISubscriptionClient,
    payload: object,
    websession: MagicMock,
) -> None:
    """Reject malformed device authorization responses."""
    websession.post.return_value = MockResponse(200, payload)

    with pytest.raises(InvalidResponseError):
        await client.async_request_device_authorization()


async def test_device_authorization_server_error(
    client: SpaceXAISubscriptionClient, websession: MagicMock
) -> None:
    """Translate an authorization endpoint server error."""
    websession.post.return_value = MockResponse(500, {})

    with pytest.raises(ConnectionFailureError):
        await client.async_request_device_authorization()


async def test_device_token_polling(
    client: SpaceXAISubscriptionClient, websession: MagicMock
) -> None:
    """Poll through authorization pending and normalize the OAuth token."""
    websession.post.side_effect = [
        MockResponse(400, {"error": "authorization_pending"}),
        MockResponse(
            200,
            {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            },
        ),
    ]
    authorization = _authorization()

    with patch(
        "spacexai_subscription_client.client.asyncio.sleep", new_callable=AsyncMock
    ):
        token = await client.async_poll_device_token(authorization)

    assert token.data["access_token"] == "access-token"
    assert token.data["refresh_token"] == "refresh-token"
    assert token.data["token_type"] == "Bearer"
    assert "expires_at" in token.data
    assert all(
        request.kwargs["headers"] == GROK_OAUTH_REQUEST_HEADERS
        for request in websession.post.call_args_list
    )


async def test_device_token_uses_server_expiry(
    client: SpaceXAISubscriptionClient, websession: MagicMock
) -> None:
    """Continue polling for the full lifetime granted by the server."""
    websession.post.side_effect = [
        MockResponse(400, {"error": "authorization_pending"}),
        MockResponse(
            200,
            {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            },
        ),
    ]

    with (
        patch(
            "spacexai_subscription_client.client.time.monotonic",
            side_effect=[0, 901, 902],
        ),
        patch(
            "spacexai_subscription_client.client.asyncio.sleep", new_callable=AsyncMock
        ),
    ):
        token = await client.async_poll_device_token(_authorization(expires_in=1800))

    assert token.data["access_token"] == "access-token"
    assert websession.post.call_count == 2


async def test_device_token_slow_down(
    client: SpaceXAISubscriptionClient, websession: MagicMock
) -> None:
    """Increase the polling delay when requested by the OAuth server."""
    websession.post.side_effect = [
        MockResponse(400, {"error": "slow_down"}),
        MockResponse(
            200,
            {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            },
        ),
    ]

    with patch(
        "spacexai_subscription_client.client.asyncio.sleep", new_callable=AsyncMock
    ) as sleep:
        await client.async_poll_device_token(_authorization())

    assert sleep.await_args_list == [call(1), call(6)]


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        pytest.param({"error": "access_denied"}, AuthorizationDeniedError, id="denied"),
        pytest.param(
            {"error": "expired_token"}, DeviceAuthorizationExpiredError, id="expired"
        ),
        pytest.param({"error": "invalid_token"}, AuthenticationError, id="invalid"),
        pytest.param({"error": "other"}, SpaceXAISubscriptionError, id="other"),
        pytest.param(ValueError(), SpaceXAISubscriptionError, id="invalid_json"),
    ],
)
async def test_device_token_error(
    client: SpaceXAISubscriptionClient,
    expected_error: type[SpaceXAISubscriptionError],
    payload: object,
    websession: MagicMock,
) -> None:
    """Translate unsuccessful token polling responses."""
    websession.post.return_value = MockResponse(400, payload)

    with pytest.raises(expected_error):
        await client.async_poll_device_token(_authorization())


@pytest.mark.parametrize(
    ("side_effect", "expected_error"),
    [
        pytest.param(TimeoutError(), RequestTimeoutError, id="timeout"),
        pytest.param(ClientError(), ConnectionFailureError, id="connection"),
    ],
)
async def test_device_token_transport_error(
    client: SpaceXAISubscriptionClient,
    expected_error: type[SpaceXAISubscriptionError],
    side_effect: Exception,
    websession: MagicMock,
) -> None:
    """Translate token polling transport failures."""
    websession.post.side_effect = side_effect

    with pytest.raises(expected_error):
        await client.async_poll_device_token(_authorization())


async def test_device_token_deadline(
    client: SpaceXAISubscriptionClient, websession: MagicMock
) -> None:
    """Stop polling when device authorization has expired."""
    with pytest.raises(DeviceAuthorizationExpiredError):
        await client.async_poll_device_token(_authorization(expires_in=0))

    websession.post.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing_token"),
        pytest.param(
            {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 0,
            },
            id="invalid_expiry",
        ),
    ],
)
async def test_device_token_invalid_success(
    client: SpaceXAISubscriptionClient,
    payload: object,
    websession: MagicMock,
) -> None:
    """Reject malformed successful token responses."""
    websession.post.return_value = MockResponse(200, payload)

    with pytest.raises(InvalidResponseError):
        await client.async_poll_device_token(_authorization())


async def test_account_authentication_error(
    client: SpaceXAISubscriptionClient, websession: MagicMock
) -> None:
    """Translate a rejected account token."""
    websession.get.return_value = MockResponse(401, {"error": "invalid_token"})

    with pytest.raises(AuthenticationError):
        await client.async_get_account("bad-token")


@pytest.mark.parametrize(
    ("status", "expected_error"),
    [
        pytest.param(401, AuthenticationError, id="authentication"),
        pytest.param(429, RateLimitError, id="rate_limit"),
        pytest.param(500, ConnectionFailureError, id="server"),
    ],
)
async def test_account_non_json_error(
    client: SpaceXAISubscriptionClient,
    expected_error: type[SpaceXAISubscriptionError],
    status: int,
    websession: MagicMock,
) -> None:
    """Classify HTTP errors even when the response body is not JSON."""
    websession.get.return_value = MockResponse(status, ValueError())

    with pytest.raises(expected_error):
        await client.async_get_account("access-token")


async def test_account(
    client: SpaceXAISubscriptionClient, websession: MagicMock
) -> None:
    """Return a normalized account identity."""
    websession.get.return_value = MockResponse(
        200, {"sub": "account-123", "email": "home@example.test"}
    )

    account = await client.async_get_account("access-token")

    assert account == Account("account-123", None, "home@example.test")
    assert account.display_name == "home@example.test"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing_subject"),
        pytest.param({"sub": "account-123", "name": 123}, id="invalid_name"),
        pytest.param(ValueError(), id="invalid_json"),
    ],
)
async def test_account_invalid_response(
    client: SpaceXAISubscriptionClient,
    payload: object,
    websession: MagicMock,
) -> None:
    """Reject malformed account responses."""
    websession.get.return_value = MockResponse(200, payload)

    with pytest.raises(InvalidResponseError):
        await client.async_get_account("access-token")


@pytest.mark.parametrize(
    ("side_effect", "expected_error"),
    [
        pytest.param(TimeoutError(), RequestTimeoutError, id="timeout"),
        pytest.param(ClientError(), ConnectionFailureError, id="connection"),
    ],
)
async def test_account_transport_error(
    client: SpaceXAISubscriptionClient,
    expected_error: type[SpaceXAISubscriptionError],
    side_effect: Exception,
    websession: MagicMock,
) -> None:
    """Translate account endpoint transport failures."""
    websession.get.side_effect = side_effect

    with pytest.raises(expected_error):
        await client.async_get_account("access-token")


async def test_models(client: SpaceXAISubscriptionClient, sdk: MagicMock) -> None:
    """Return a sorted model catalog."""
    sdk.models.list.return_value = MagicMock(
        data=[MagicMock(id="grok-4.6"), MagicMock(id="grok-4.5")]
    )

    models = await client.async_list_models("access-token")

    assert models == ("grok-4.5", "grok-4.6")
    sdk.models.list.assert_awaited_once_with(timeout=MODEL_CATALOG_TIMEOUT)
    assert sdk.constructor.call_args.kwargs["max_retries"] == SDK_MAX_RETRIES
    assert sdk.constructor.call_args.kwargs["default_headers"] == {
        **GROK_CLI_REQUEST_HEADERS,
        "User-Agent": "spacexai-subscription-client/0.2.0",
        "x-grok-client-version": "0.2.0",
    }


@pytest.mark.parametrize(("sdk_error", "expected_error"), SDK_ERRORS)
async def test_model_error(
    client: SpaceXAISubscriptionClient,
    expected_error: type[SpaceXAISubscriptionError],
    sdk: MagicMock,
    sdk_error: openai.OpenAIError,
) -> None:
    """Translate model catalog SDK failures."""
    sdk.models.list.side_effect = sdk_error

    with pytest.raises(expected_error):
        await client.async_list_models("access-token")


async def test_invalid_model_catalog(
    client: SpaceXAISubscriptionClient, sdk: MagicMock
) -> None:
    """Reject an invalid model catalog."""
    sdk.models.list.return_value = MagicMock(data=[MagicMock(id=None)])

    with pytest.raises(InvalidResponseError):
        await client.async_list_models("access-token")


async def test_completion(client: SpaceXAISubscriptionClient, sdk: MagicMock) -> None:
    """Normalize text and client-side tool calls."""
    response = MagicMock()
    response.output_text = "Calling a tool"
    response.output = [
        ResponseFunctionToolCall(
            arguments='{"area":"kitchen"}',
            call_id="call-1",
            name="HassTurnOn",
            type="function_call",
        )
    ]
    sdk.responses.create.return_value = response

    completion = await client.async_create_response(
        "access-token",
        model="grok-4.6",
        input_data=[Message("user", "Turn on the kitchen")],
        tools=[
            Tool(
                "HassTurnOn",
                "Turn on a device",
                {"type": "object", "properties": {}},
            )
        ],
    )

    assert completion.text == "Calling a tool"
    assert completion.tool_calls[0].name == "HassTurnOn"
    assert completion.tool_calls[0].arguments == {"area": "kitchen"}
    request: dict[str, Any] = sdk.responses.create.call_args.kwargs
    assert request["model"] == "grok-4.6"
    assert request["parallel_tool_calls"] is False
    assert request["extra_headers"] == {"x-grok-model-override": "grok-4.6"}
    assert request["timeout"] == RESPONSE_TIMEOUT
    assert sdk.constructor.call_args.kwargs["max_retries"] == SDK_MAX_RETRIES


async def test_completion_formats_tool_history(
    client: SpaceXAISubscriptionClient, sdk: MagicMock
) -> None:
    """Format prior tool calls and results for the SDK request."""
    response = MagicMock(output=[], output_text="Done")
    sdk.responses.create.return_value = response

    await client.async_create_response(
        "old-access-token",
        model="grok-4.6",
        input_data=[],
        tools=[],
    )
    await client.async_create_response(
        "new-access-token",
        model="grok-4.6",
        input_data=[
            ToolCall("call-1", "HassTurnOn", {"area": "kitchen"}),
            ToolResult("call-1", '"done"'),
        ],
        tools=[],
    )

    request: dict[str, Any] = sdk.responses.create.call_args.kwargs
    assert request["input"] == [
        {
            "type": "function_call",
            "name": "HassTurnOn",
            "arguments": '{"area": "kitchen"}',
            "call_id": "call-1",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '"done"',
        },
    ]
    assert sdk.constructor.call_args_list[-1].kwargs["api_key"] == "new-access-token"


async def test_completion_formats_attachments(
    client: SpaceXAISubscriptionClient, sdk: MagicMock
) -> None:
    """Format image and PDF attachments on a user message."""
    sdk.responses.create.return_value = MagicMock(output=[], output_text="Done")

    await client.async_create_response(
        "access-token",
        model="grok-4.6",
        input_data=[
            Message(
                "user",
                "Compare these files",
                (
                    Attachment("image.png", "image/png", b"image"),
                    Attachment("document.pdf", "application/pdf", b"pdf"),
                ),
            )
        ],
        tools=[],
    )

    request: dict[str, Any] = sdk.responses.create.call_args.kwargs
    assert request["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Compare these files"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aW1hZ2U=",
                    "detail": "auto",
                },
                {
                    "type": "input_file",
                    "filename": "document.pdf",
                    "file_data": "data:application/pdf;base64,cGRm",
                },
            ],
        }
    ]


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            Message("user", "", (Attachment("", "image/png", b"image"),)),
            id="missing_filename",
        ),
        pytest.param(
            Message("user", "", (Attachment("image.png", "image/png", b""),)),
            id="empty_data",
        ),
        pytest.param(
            Message("user", "", (Attachment("data.txt", "text/plain", b"data"),)),
            id="unsupported_type",
        ),
    ],
)
async def test_completion_rejects_invalid_attachment(
    client: SpaceXAISubscriptionClient, message: Message, sdk: MagicMock
) -> None:
    """Reject invalid attachment input before requesting a response."""
    with pytest.raises(
        ValueError, match=r"Attachments require|Unsupported attachment type"
    ):
        await client.async_create_response(
            "access-token",
            model="grok-4.6",
            input_data=[message],
            tools=[],
        )
    sdk.responses.create.assert_not_awaited()


def test_message_rejects_assistant_attachment() -> None:
    """Reject attachments on non-user messages."""
    with pytest.raises(ValueError, match="only valid on user messages"):
        Message(
            "assistant",
            "Attached",
            (Attachment("image.png", "image/png", b"image"),),
        )


async def test_completion_formats_builtin_tools(
    client: SpaceXAISubscriptionClient, sdk: MagicMock
) -> None:
    """Format provider-hosted tools for the SDK request."""
    sdk.responses.create.return_value = MagicMock(output=[], output_text="Done")

    await client.async_create_response(
        "access-token",
        model="grok-4.6",
        input_data=[Message("user", "Search for this")],
        tools=[
            BuiltinTool("web_search"),
            BuiltinTool("x_search"),
            BuiltinTool("code_interpreter"),
        ],
    )

    request: dict[str, Any] = sdk.responses.create.call_args.kwargs
    assert request["tools"] == [
        {"type": "web_search"},
        {"type": "x_search"},
        {"type": "code_interpreter"},
    ]


@pytest.mark.parametrize(("sdk_error", "expected_error"), SDK_ERRORS)
async def test_completion_error(
    client: SpaceXAISubscriptionClient,
    expected_error: type[SpaceXAISubscriptionError],
    sdk: MagicMock,
    sdk_error: openai.OpenAIError,
) -> None:
    """Translate completion SDK failures."""
    sdk.responses.create.side_effect = sdk_error

    with pytest.raises(expected_error):
        await client.async_create_response(
            "access-token",
            model="grok-4.6",
            input_data=[],
            tools=[],
        )


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(MagicMock(output=[], output_text=""), id="empty"),
        pytest.param(
            MagicMock(
                output=[
                    ResponseFunctionToolCall(
                        arguments="not-json",
                        call_id="call-1",
                        name="HassTurnOn",
                        type="function_call",
                    )
                ],
                output_text="",
            ),
            id="invalid_json",
        ),
        pytest.param(
            MagicMock(
                output=[
                    ResponseFunctionToolCall(
                        arguments="[]",
                        call_id="call-1",
                        name="HassTurnOn",
                        type="function_call",
                    )
                ],
                output_text="",
            ),
            id="invalid_arguments",
        ),
        pytest.param(
            MagicMock(
                output=[
                    ResponseFunctionToolCall(
                        arguments="{}",
                        call_id="",
                        name="HassTurnOn",
                        type="function_call",
                    )
                ],
                output_text="",
            ),
            id="missing_call_id",
        ),
        pytest.param(
            MagicMock(
                output=[
                    ResponseFunctionToolCall(
                        arguments="{}",
                        call_id="call-1",
                        name="",
                        type="function_call",
                    )
                ],
                output_text="",
            ),
            id="missing_tool_name",
        ),
    ],
)
async def test_completion_invalid_response(
    client: SpaceXAISubscriptionClient,
    response: MagicMock,
    sdk: MagicMock,
) -> None:
    """Reject malformed completion responses."""
    sdk.responses.create.return_value = response

    with pytest.raises(InvalidResponseError):
        await client.async_create_response(
            "access-token",
            model="grok-4.6",
            input_data=[],
            tools=[],
        )
