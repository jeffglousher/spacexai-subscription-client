"""Async client for SpaceXAI OAuth subscription APIs."""

import asyncio
import base64
import binascii
import json
import time
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import openai
from aiohttp import ClientError, ClientResponse, ClientSession, ClientTimeout, FormData
from httpx import AsyncClient as HttpxClient
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseTextConfigParam,
    ToolParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from .const import (
    API_BASE_URL,
    DEVICE_CODE_GRANT,
    DEVICE_CODE_URL,
    GROK_CLI_OAUTH_CLIENT_ID,
    GROK_CLI_REQUEST_HEADERS,
    GROK_OAUTH_REQUEST_HEADERS,
    HTTP_TIMEOUT,
    IMAGE_TIMEOUT,
    IMAGES_EDIT_URL,
    IMAGES_URL,
    MAX_IMAGE_SIZE,
    MAX_STT_SIZE,
    MAX_TTS_SIZE,
    MODEL_CATALOG_TIMEOUT,
    OAUTH_REFERRER,
    OAUTH_SCOPES,
    RESPONSE_TIMEOUT,
    SDK_MAX_RETRIES,
    SPEECH_TIMEOUT,
    STT_URL,
    TOKEN_URL,
    TTS_MAX_SPEED,
    TTS_MAX_TEXT_LENGTH,
    TTS_MIN_SPEED,
    TTS_URL,
    USERINFO_URL,
    VIDEO_GENERATIONS_URL,
    VIDEO_MAX_DURATION,
    VIDEO_POLL_INTERVAL,
    VIDEO_POLL_TIMEOUT,
    VIDEO_REQUEST_TIMEOUT,
    VIDEOS_URL,
)
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
    Attachment,
    BuiltinTool,
    Completion,
    DeviceAuthorization,
    GeneratedImage,
    GeneratedVideo,
    InputItem,
    Message,
    OAuthToken,
    ResponseFormat,
    ResponseTool,
    ToolCall,
)

_TIMEOUT = ClientTimeout(total=HTTP_TIMEOUT)
_MAX_EDIT_IMAGES = 5
_WEBP_HEADER_SIZE = 12
_VIDEO_ASPECT_RATIOS = frozenset({"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"})
_VIDEO_RESOLUTIONS = frozenset({"480p", "720p", "1080p"})


class SpaceXAISubscriptionClient:
    """Access SpaceXAI OAuth and subscription endpoints."""

    def __init__(self, websession: ClientSession, http_client: HttpxClient) -> None:
        """Initialize the client with caller-owned HTTP sessions."""
        self._websession = websession
        self._http_client = http_client

    async def async_request_device_authorization(self) -> DeviceAuthorization:
        """Start OAuth device authorization."""
        try:
            async with self._websession.post(
                DEVICE_CODE_URL,
                data={
                    "client_id": GROK_CLI_OAUTH_CLIENT_ID,
                    "referrer": OAUTH_REFERRER,
                    "scope": " ".join(OAUTH_SCOPES),
                },
                headers=GROK_OAUTH_REQUEST_HEADERS,
                timeout=_TIMEOUT,
            ) as response:
                payload = await _async_json(response)
                _raise_for_status(response.status, payload)
        except SpaceXAISubscriptionError:
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
        deadline = time.monotonic() + authorization.expires_in
        interval = authorization.interval

        while True:
            await asyncio.sleep(interval)
            if time.monotonic() > deadline:
                raise DeviceAuthorizationExpiredError
            try:
                async with self._websession.post(
                    TOKEN_URL,
                    data={
                        "grant_type": DEVICE_CODE_GRANT,
                        "client_id": GROK_CLI_OAUTH_CLIENT_ID,
                        "device_code": authorization.device_code,
                    },
                    headers=GROK_OAUTH_REQUEST_HEADERS,
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

            interval = _next_poll_interval(response.status, payload, interval)

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
        except SpaceXAISubscriptionError:
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
            models = await self._sdk(access_token).models.list(
                timeout=MODEL_CATALOG_TIMEOUT
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
            raise SpaceXAISubscriptionError from err
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
        tools: Sequence[ResponseTool],
        response_format: ResponseFormat | None = None,
    ) -> Completion:
        """Create a non-streaming Grok response."""
        try:
            response = await self._sdk(access_token).responses.create(
                model=model,
                input=_format_input(input_data),
                tools=[_format_tool(tool) for tool in tools],
                parallel_tool_calls=False,
                text=_format_response_text(response_format),
                extra_headers={"x-grok-model-override": model},
                timeout=RESPONSE_TIMEOUT,
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
            raise SpaceXAISubscriptionError from err
        tool_calls = tuple(
            _parse_tool_call(item)
            for item in response.output
            if isinstance(item, ResponseFunctionToolCall)
        )
        if not response.output_text and not tool_calls:
            raise InvalidResponseError
        return Completion(response.output_text or "", tool_calls)

    async def async_generate_image(
        self,
        access_token: str,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
    ) -> GeneratedImage:
        """Generate an image from a prompt."""
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
        if aspect_ratio is not None:
            body["aspect_ratio"] = aspect_ratio
        if resolution is not None:
            body["resolution"] = resolution
        payload = await self._async_image_request(access_token, IMAGES_URL, body)
        return _parse_generated_image(payload, model)

    async def async_edit_image(
        self,
        access_token: str,
        *,
        model: str,
        prompt: str,
        images: Sequence[Attachment],
        aspect_ratio: str | None = None,
    ) -> GeneratedImage:
        """Edit an image using up to five references."""
        if not 1 <= len(images) <= _MAX_EDIT_IMAGES:
            message = "Image editing requires between one and five images"
            raise ValueError(message)
        image_objects = [
            {"type": "image_url", "url": _format_image_data_url(image)}
            for image in images
        ]
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json",
        }
        body["image" if len(image_objects) == 1 else "images"] = (
            image_objects[0] if len(image_objects) == 1 else image_objects
        )
        if aspect_ratio is not None:
            body["aspect_ratio"] = aspect_ratio
        payload = await self._async_image_request(access_token, IMAGES_EDIT_URL, body)
        return _parse_generated_image(payload, model)

    async def _async_image_request(
        self, access_token: str, url: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Request an image operation and return its JSON payload."""
        try:
            async with self._websession.post(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                json=body,
                timeout=ClientTimeout(total=IMAGE_TIMEOUT),
            ) as response:
                payload = await _async_json(response)
                _raise_for_status(response.status, payload)
        except SpaceXAISubscriptionError:
            raise
        except TimeoutError as err:
            raise RequestTimeoutError from err
        except ClientError as err:
            raise ConnectionFailureError from err
        return payload

    async def async_generate_video(  # noqa: PLR0913
        self,
        access_token: str,
        *,
        model: str,
        prompt: str,
        image_url: str | None = None,
        duration: int | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
    ) -> GeneratedVideo:
        """Generate a video and wait for the deferred result."""
        payload = await self._async_video_request(
            access_token,
            VIDEO_GENERATIONS_URL,
            body=_format_video_request(
                model, prompt, image_url, duration, aspect_ratio, resolution
            ),
        )
        try:
            request_id = _required_string(payload, "request_id")
        except (KeyError, TypeError) as err:
            raise InvalidResponseError from err
        return await self._async_poll_video(access_token, request_id, model)

    async def _async_poll_video(
        self, access_token: str, request_id: str, requested_model: str
    ) -> GeneratedVideo:
        """Poll a deferred video request until it reaches a terminal state."""
        try:
            async with asyncio.timeout(VIDEO_POLL_TIMEOUT):
                while True:
                    payload = await self._async_video_request(
                        access_token, f"{VIDEOS_URL}/{request_id}"
                    )
                    status = _video_status(payload)
                    if status == "done":
                        return _parse_generated_video(payload, requested_model)
                    if status in ("failed", "expired"):
                        raise SpaceXAISubscriptionError
                    await asyncio.sleep(VIDEO_POLL_INTERVAL)
        except TimeoutError as err:
            raise RequestTimeoutError from err

    async def _async_video_request(
        self,
        access_token: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request a video operation and return its JSON payload."""
        try:
            request = (
                self._websession.post if body is not None else self._websession.get
            )
            kwargs: dict[str, Any] = {
                "headers": {"Authorization": f"Bearer {access_token}"},
                "timeout": ClientTimeout(total=VIDEO_REQUEST_TIMEOUT),
            }
            if body is not None:
                kwargs["json"] = body
            async with request(url, **kwargs) as response:
                payload = await _async_json(response)
                _raise_for_status(response.status, payload)
        except SpaceXAISubscriptionError:
            raise
        except TimeoutError as err:
            raise RequestTimeoutError from err
        except ClientError as err:
            raise ConnectionFailureError from err
        return payload

    async def async_transcribe(
        self,
        access_token: str,
        *,
        audio: bytes,
        filename: str,
        media_type: str,
        language: str | None = None,
    ) -> str:
        """Transcribe an audio file."""
        if not audio or len(audio) > MAX_STT_SIZE:
            message = f"Transcription audio must contain 1-{MAX_STT_SIZE} bytes"
            raise ValueError(message)
        if not filename or not media_type:
            message = "Transcription filename and media type cannot be empty"
            raise ValueError(message)
        if language == "":
            message = "Transcription language cannot be empty"
            raise ValueError(message)
        form = FormData()
        if language is not None:
            form.add_field("format", "true")
            form.add_field("language", language)
        form.add_field("file", audio, filename=filename, content_type=media_type)
        try:
            async with self._websession.post(
                STT_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                data=form,
                timeout=ClientTimeout(total=SPEECH_TIMEOUT),
            ) as response:
                payload = await _async_json(response)
                _raise_for_status(response.status, payload)
        except SpaceXAISubscriptionError:
            raise
        except TimeoutError as err:
            raise RequestTimeoutError from err
        except ClientError as err:
            raise ConnectionFailureError from err
        try:
            return _required_string(payload, "text")
        except (KeyError, TypeError) as err:
            raise InvalidResponseError from err

    async def async_synthesize_speech(  # noqa: PLR0913
        self,
        access_token: str,
        *,
        text: str,
        voice_id: str,
        language: str,
        speed: float = 1.0,
        codec: Literal["mp3", "pcm", "wav"] = "mp3",
    ) -> bytes:
        """Synthesize speech as raw audio bytes."""
        if not text or len(text) > TTS_MAX_TEXT_LENGTH:
            message = f"Speech text must contain 1-{TTS_MAX_TEXT_LENGTH} characters"
            raise ValueError(message)
        if not voice_id or not language:
            message = "Speech voice and language cannot be empty"
            raise ValueError(message)
        if not TTS_MIN_SPEED <= speed <= TTS_MAX_SPEED:
            message = (
                f"Speech speed must be between {TTS_MIN_SPEED} and {TTS_MAX_SPEED}"
            )
            raise ValueError(message)
        if codec not in ("mp3", "pcm", "wav"):
            message = f"Unsupported speech codec: {codec}"
            raise ValueError(message)
        try:
            async with self._websession.post(
                TTS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                json={
                    "text": text,
                    "voice_id": voice_id,
                    "language": language,
                    "speed": speed,
                    "output_format": {"codec": codec},
                },
                timeout=ClientTimeout(total=SPEECH_TIMEOUT),
            ) as response:
                if response.status >= HTTPStatus.BAD_REQUEST:
                    _raise_for_status(response.status, await _async_json(response))
                audio = await response.content.read(MAX_TTS_SIZE + 1)
        except SpaceXAISubscriptionError:
            raise
        except TimeoutError as err:
            raise RequestTimeoutError from err
        except ClientError as err:
            raise ConnectionFailureError from err
        if not audio or len(audio) > MAX_TTS_SIZE:
            raise InvalidResponseError
        return audio

    def _sdk(self, access_token: str) -> openai.AsyncOpenAI:
        """Return a request-scoped SDK client with the current access token."""
        return openai.AsyncOpenAI(
            api_key=access_token,
            base_url=API_BASE_URL,
            default_headers=GROK_CLI_REQUEST_HEADERS,
            http_client=self._http_client,
            max_retries=SDK_MAX_RETRIES,
        )


def _format_input(items: Sequence[InputItem]) -> ResponseInputParam:
    """Convert normalized input into SDK request items."""
    result: ResponseInputParam = []
    for item in items:
        if isinstance(item, Message):
            content: (
                str
                | list[
                    ResponseInputTextParam
                    | ResponseInputImageParam
                    | ResponseInputFileParam
                ]
            ) = item.content
            if item.attachments:
                content = []
                if item.content:
                    content.append(
                        ResponseInputTextParam(type="input_text", text=item.content)
                    )
                content.extend(
                    _format_attachment(attachment) for attachment in item.attachments
                )
            result.append(
                EasyInputMessageParam(
                    type="message",
                    role=item.role,
                    content=content,
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


def _format_video_request(  # noqa: PLR0913, PLR0917
    model: str,
    prompt: str,
    image_url: str | None,
    duration: int | None,
    aspect_ratio: str | None,
    resolution: str | None,
) -> dict[str, Any]:
    """Validate and format a video generation request."""
    if not model or not prompt:
        message = "Video model and prompt cannot be empty"
        raise ValueError(message)
    if image_url == "":
        message = "Video input image URL cannot be empty"
        raise ValueError(message)
    if duration is not None and not 1 <= duration <= VIDEO_MAX_DURATION:
        message = f"Video duration must be between 1 and {VIDEO_MAX_DURATION} seconds"
        raise ValueError(message)
    if aspect_ratio is not None and aspect_ratio not in _VIDEO_ASPECT_RATIOS:
        message = f"Unsupported video aspect ratio: {aspect_ratio}"
        raise ValueError(message)
    if resolution is not None and resolution not in _VIDEO_RESOLUTIONS:
        message = f"Unsupported video resolution: {resolution}"
        raise ValueError(message)
    body: dict[str, Any] = {"model": model, "prompt": prompt}
    body.update(
        {
            key: value
            for key, value in (
                ("duration", duration),
                ("aspect_ratio", aspect_ratio),
                ("resolution", resolution),
            )
            if value is not None
        }
    )
    if image_url is not None:
        body["image"] = {"url": image_url}
    return body


def _video_status(payload: dict[str, Any]) -> str:
    """Validate a deferred video status."""
    try:
        status = _required_string(payload, "status")
    except (KeyError, TypeError) as err:
        raise InvalidResponseError from err
    if status not in ("pending", "done", "failed", "expired"):
        raise InvalidResponseError
    return status


def _format_attachment(
    attachment: Attachment,
) -> ResponseInputImageParam | ResponseInputFileParam:
    """Convert a binary attachment to an SDK input part."""
    if not attachment.filename or not attachment.data:
        message = "Attachments require a filename and data"
        raise ValueError(message)
    encoded = base64.b64encode(attachment.data).decode()
    data_url = f"data:{attachment.media_type};base64,{encoded}"
    if attachment.media_type.startswith("image/"):
        return ResponseInputImageParam(
            type="input_image",
            image_url=data_url,
            detail="auto",
        )
    if attachment.media_type == "application/pdf":
        return ResponseInputFileParam(
            type="input_file",
            filename=attachment.filename,
            file_data=data_url,
        )
    message = f"Unsupported attachment type: {attachment.media_type}"
    raise ValueError(message)


def _format_image_data_url(attachment: Attachment) -> str:
    """Convert an image attachment to a data URL."""
    if attachment.media_type not in ("image/jpeg", "image/png", "image/webp"):
        message = f"Unsupported edit image type: {attachment.media_type}"
        raise ValueError(message)
    if not attachment.data:
        message = "Edit images require data"
        raise ValueError(message)
    encoded = base64.b64encode(attachment.data).decode()
    return f"data:{attachment.media_type};base64,{encoded}"


def _format_response_text(
    response_format: ResponseFormat | None,
) -> ResponseTextConfigParam | openai.Omit:
    """Convert an optional structured response format."""
    if response_format is None:
        return openai.omit
    if not response_format.name:
        message = "Structured responses require a name"
        raise ValueError(message)
    return ResponseTextConfigParam(
        format=ResponseFormatTextJSONSchemaConfigParam(
            type="json_schema",
            name=response_format.name,
            schema=dict(response_format.schema),
            strict=True,
        )
    )


def _format_tool(tool: ResponseTool) -> ToolParam:
    """Convert a normalized tool to an SDK request object."""
    if isinstance(tool, BuiltinTool):
        return cast("ToolParam", {"type": tool.type})
    return FunctionToolParam(
        type="function",
        name=tool.name,
        description=tool.description,
        parameters=dict(tool.parameters),
        strict=False,
    )


def _parse_tool_call(item: ResponseFunctionToolCall) -> ToolCall:
    """Convert and validate an SDK tool call."""
    if not item.call_id or not item.name:
        raise InvalidResponseError
    try:
        arguments = json.loads(item.arguments)
    except ValueError as err:
        raise InvalidResponseError from err
    if not isinstance(arguments, dict):
        raise InvalidResponseError
    return ToolCall(item.call_id, item.name, arguments)


def _parse_generated_image(payload: object, model: str) -> GeneratedImage:
    """Validate and normalize an Imagine API response."""
    if not isinstance(payload, Mapping):
        raise InvalidResponseError
    entries = payload.get("data")
    if not isinstance(entries, list) or not entries:
        raise InvalidResponseError
    first = entries[0]
    if not isinstance(first, Mapping):
        raise InvalidResponseError
    encoded = first.get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise InvalidResponseError
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as err:
        raise InvalidResponseError from err
    if not data or len(data) > MAX_IMAGE_SIZE:
        raise InvalidResponseError
    revised_prompt = first.get("revised_prompt")
    if revised_prompt is not None and not isinstance(revised_prompt, str):
        raise InvalidResponseError
    return GeneratedImage(
        data,
        _image_media_type(data, first.get("mime_type")),
        model,
        revised_prompt,
    )


def _parse_generated_video(payload: object, requested_model: str) -> GeneratedVideo:
    """Validate and normalize a completed Imagine video response."""
    if not isinstance(payload, Mapping):
        raise InvalidResponseError
    video = payload.get("video")
    if not isinstance(video, Mapping):
        raise InvalidResponseError
    url = video.get("url")
    duration = video.get("duration")
    respect_moderation = video.get("respect_moderation")
    model = payload.get("model", requested_model)
    if (
        not isinstance(url, str)
        or urlsplit(url).scheme != "https"
        or not urlsplit(url).netloc
        or not isinstance(duration, int)
        or isinstance(duration, bool)
        or duration <= 0
        or not isinstance(respect_moderation, bool)
        or not isinstance(model, str)
        or not model
    ):
        raise InvalidResponseError
    return GeneratedVideo(url, model, duration, respect_moderation)


def _image_media_type(data: bytes, provider_type: object) -> str:
    """Return the image media type from magic bytes or provider metadata."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if (
        len(data) >= _WEBP_HEADER_SIZE
        and data.startswith(b"RIFF")
        and data[8:12] == b"WEBP"
    ):
        return "image/webp"
    if isinstance(provider_type, str) and provider_type.startswith("image/"):
        return provider_type
    raise InvalidResponseError


async def _async_json(response: ClientResponse) -> dict[str, Any]:
    """Decode a provider JSON object."""
    try:
        payload = await response.json(content_type=None)
    except ValueError as err:
        _raise_for_status(response.status, {})
        raise InvalidResponseError from err
    if not isinstance(payload, dict):
        _raise_for_status(response.status, {})
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


def _next_poll_interval(status: int, payload: dict[str, Any], interval: int) -> int:
    """Handle a device-token polling response and return the next interval."""
    error = payload.get("error")
    if error == "authorization_pending":
        return interval
    if error == "slow_down":
        return min(interval + 5, 30)
    if error in ("access_denied", "authorization_denied"):
        raise AuthorizationDeniedError
    if error == "expired_token":
        raise DeviceAuthorizationExpiredError
    _raise_for_status(status, payload)
    return interval


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
    raise SpaceXAISubscriptionError
