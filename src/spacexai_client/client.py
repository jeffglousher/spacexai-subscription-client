"""Async SpaceXAI client."""

import asyncio
import json
import time
from collections.abc import Sequence
from http import HTTPStatus
from typing import Any

import openai
from aiohttp import ClientError, ClientResponse, ClientSession, ClientTimeout
from httpx import AsyncClient as HttpxClient
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from .const import (
    API_BASE_URL,
    DEVICE_CODE_GRANT,
    DEVICE_CODE_MAX_POLL_SECONDS,
    DEVICE_CODE_URL,
    GROK_CLI_OAUTH_CLIENT_ID,
    GROK_CLI_REQUEST_HEADERS,
    HTTP_TIMEOUT,
    OAUTH_SCOPES,
    TOKEN_URL,
    USERINFO_URL,
)
from .errors import (
    AuthenticationError,
    AuthorizationDeniedError,
    ConnectionFailureError,
    InvalidResponseError,
    RateLimitError,
    RequestTimeoutError,
    SpaceXAIError,
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
)

_TIMEOUT = ClientTimeout(total=HTTP_TIMEOUT)


class SpaceXAIClient:
    """Access SpaceXAI OAuth and Grok subscription endpoints."""

    def __init__(self, websession: ClientSession, http_client: HttpxClient) -> None:
        """Initialize the client with caller-owned HTTP sessions."""
        self._websession = websession
        self._http_client = http_client
        self._sdk_client: openai.AsyncOpenAI | None = None

    async def async_request_device_authorization(self) -> DeviceAuthorization:
        """Start OAuth device authorization."""
        try:
            async with self._websession.post(
                DEVICE_CODE_URL,
                data={
                    "client_id": GROK_CLI_OAUTH_CLIENT_ID,
                    "scope": " ".join(OAUTH_SCOPES),
                },
                timeout=_TIMEOUT,
            ) as response:
                payload = await _async_json(response)
                _raise_for_status(response.status, payload)
        except SpaceXAIError:
            raise
        except TimeoutError as err:
            raise RequestTimeoutError from err
        except ClientError as err:
            raise ConnectionFailureError from err

        try:
            verification_uri = _required_string(payload, "verification_uri")
            expires_in = int(payload["expires_in"])
            interval = max(1, int(payload.get("interval", 5)))
            device_code = _required_string(payload, "device_code")
            user_code = _required_string(payload, "user_code")
        except (KeyError, TypeError, ValueError) as err:
            raise InvalidResponseError from err
        verification_uri_complete = payload.get(
            "verification_uri_complete", verification_uri
        )
        if (
            not isinstance(verification_uri_complete, str)
            or not verification_uri_complete
            or expires_in <= 0
        ):
            raise InvalidResponseError
        return DeviceAuthorization(
            device_code=device_code,
            user_code=user_code,
            verification_uri=verification_uri,
            verification_uri_complete=verification_uri_complete,
            expires_in=expires_in,
            interval=interval,
        )

    async def async_poll_device_token(
        self, authorization: DeviceAuthorization
    ) -> OAuthToken:
        """Poll until the user approves device authorization."""
        deadline = time.monotonic() + min(
            authorization.expires_in, DEVICE_CODE_MAX_POLL_SECONDS
        )
        interval = authorization.interval

        while time.monotonic() < deadline:
            try:
                async with self._websession.post(
                    TOKEN_URL,
                    data={
                        "grant_type": DEVICE_CODE_GRANT,
                        "client_id": GROK_CLI_OAUTH_CLIENT_ID,
                        "device_code": authorization.device_code,
                    },
                    timeout=_TIMEOUT,
                ) as response:
                    payload = await _async_json(response)
            except InvalidResponseError:
                raise
            except TimeoutError as err:
                raise RequestTimeoutError from err
            except ClientError as err:
                raise ConnectionFailureError from err

            if response.status == HTTPStatus.OK:
                return _oauth_token(payload)

            error = payload.get("error")
            if error == "authorization_pending":
                await asyncio.sleep(interval)
                continue
            if error == "slow_down":
                interval = min(interval + 5, 30)
                await asyncio.sleep(interval)
                continue
            if error in ("access_denied", "authorization_denied"):
                raise AuthorizationDeniedError
            if error == "expired_token":
                raise RequestTimeoutError
            _raise_for_status(response.status, payload)

        raise RequestTimeoutError

    async def async_get_account(self, access_token: str) -> Account:
        """Return the authenticated account identity."""
        try:
            async with self._websession.get(
                USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=_TIMEOUT,
            ) as response:
                payload = await _async_json(response)
                _raise_for_status(response.status, payload)
        except SpaceXAIError:
            raise
        except TimeoutError as err:
            raise RequestTimeoutError from err
        except ClientError as err:
            raise ConnectionFailureError from err

        subject = payload.get("sub")
        name = payload.get("name")
        email = payload.get("email")
        if (
            not isinstance(subject, str)
            or not subject
            or (name is not None and not isinstance(name, str))
            or (email is not None and not isinstance(email, str))
        ):
            raise InvalidResponseError
        return Account(subject, name, email)

    async def async_list_models(self, access_token: str) -> tuple[str, ...]:
        """Return model identifiers available to the OAuth account."""
        try:
            models = await self._sdk(access_token).models.list(timeout=10.0)
        except openai.AuthenticationError as err:
            raise AuthenticationError from err
        except openai.APITimeoutError as err:
            raise RequestTimeoutError from err
        except openai.APIConnectionError as err:
            raise ConnectionFailureError from err
        except openai.RateLimitError as err:
            raise RateLimitError from err
        except openai.OpenAIError as err:
            raise SpaceXAIError from err
        try:
            model_ids = tuple(model.id for model in models.data)
        except (AttributeError, TypeError) as err:
            raise InvalidResponseError from err
        if any(not isinstance(model_id, str) or not model_id for model_id in model_ids):
            raise InvalidResponseError
        return tuple(sorted(model_ids))

    async def async_create_response(
        self,
        access_token: str,
        *,
        model: str,
        input_data: Sequence[InputItem],
        tools: Sequence[Tool],
    ) -> Completion:
        """Create a non-streaming Grok response."""
        try:
            response = await self._sdk(access_token).responses.create(
                model=model,
                input=_format_input(input_data),
                tools=[
                    FunctionToolParam(
                        type="function",
                        name=tool.name,
                        description=tool.description,
                        parameters=dict(tool.parameters),
                        strict=False,
                    )
                    for tool in tools
                ],
                parallel_tool_calls=False,
                extra_headers={"x-grok-model-override": model},
            )
        except openai.AuthenticationError as err:
            raise AuthenticationError from err
        except openai.APITimeoutError as err:
            raise RequestTimeoutError from err
        except openai.APIConnectionError as err:
            raise ConnectionFailureError from err
        except openai.RateLimitError as err:
            raise RateLimitError from err
        except openai.OpenAIError as err:
            raise SpaceXAIError from err
        tool_calls = tuple(
            _parse_tool_call(item)
            for item in response.output
            if isinstance(item, ResponseFunctionToolCall)
        )
        if not response.output_text and not tool_calls:
            raise InvalidResponseError
        return Completion(response.output_text or "", tool_calls)

    def _sdk(self, access_token: str) -> openai.AsyncOpenAI:
        """Return the shared SDK client with the current access token."""
        if self._sdk_client is None:
            self._sdk_client = openai.AsyncOpenAI(
                api_key=access_token,
                base_url=API_BASE_URL,
                default_headers=GROK_CLI_REQUEST_HEADERS,
                http_client=self._http_client,
            )
        else:
            self._sdk_client.api_key = access_token
        return self._sdk_client


def _format_input(items: Sequence[InputItem]) -> ResponseInputParam:
    """Convert normalized input into SDK request items."""
    result: ResponseInputParam = []
    for item in items:
        if isinstance(item, Message):
            result.append(
                EasyInputMessageParam(
                    type="message",
                    role=item.role,
                    content=item.content,
                )
            )
        elif isinstance(item, ToolCall):
            result.append(
                ResponseFunctionToolCallParam(
                    type="function_call",
                    name=item.name,
                    arguments=json.dumps(dict(item.arguments)),
                    call_id=item.id,
                )
            )
        else:
            result.append(
                FunctionCallOutput(
                    type="function_call_output",
                    call_id=item.call_id,
                    output=item.output,
                )
            )
    return result


def _parse_tool_call(item: ResponseFunctionToolCall) -> ToolCall:
    """Convert and validate an SDK tool call."""
    try:
        arguments = json.loads(item.arguments)
    except ValueError as err:
        raise InvalidResponseError from err
    if not isinstance(arguments, dict):
        raise InvalidResponseError
    return ToolCall(item.call_id, item.name, arguments)


async def _async_json(response: ClientResponse) -> dict[str, Any]:
    """Decode a provider JSON object."""
    try:
        payload = await response.json(content_type=None)
    except ValueError as err:
        raise InvalidResponseError from err
    if not isinstance(payload, dict):
        raise InvalidResponseError
    return payload


def _required_string(payload: dict[str, Any], key: str) -> str:
    """Return a required non-empty string."""
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise TypeError
    return value


def _oauth_token(payload: dict[str, Any]) -> OAuthToken:
    """Validate and normalize a token response."""
    try:
        access_token = _required_string(payload, "access_token")
        refresh_token = _required_string(payload, "refresh_token")
        expires_in = int(payload["expires_in"])
    except (KeyError, TypeError, ValueError) as err:
        raise InvalidResponseError from err
    if expires_in <= 0:
        raise InvalidResponseError
    token = dict(payload)
    token.update(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        expires_at=time.time() + expires_in,
    )
    token.setdefault("token_type", "Bearer")
    return OAuthToken(token)


def _raise_for_status(status: int, payload: dict[str, Any]) -> None:
    """Translate an HTTP status into a stable client exception."""
    if status < HTTPStatus.BAD_REQUEST:
        return
    if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN) or payload.get(
        "error"
    ) in (
        "invalid_client",
        "invalid_token",
        "unauthorized_client",
    ):
        raise AuthenticationError
    if status == HTTPStatus.REQUEST_TIMEOUT:
        raise RequestTimeoutError
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        raise RateLimitError
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        raise ConnectionFailureError
    raise SpaceXAIError
