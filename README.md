# spacexai-subscription-client

`spacexai-subscription-client` is a small, typed async client for SpaceXAI OAuth
device login and subscription inference. It is designed for applications such
as Home Assistant that own OAuth token storage and refresh.

This is an unofficial community package, not xAI's official Python SDK.
Authentication uses the SpaceXAI OAuth flow; API-key authentication is outside
this package's scope. Internally, the OAuth access token is supplied to the
OpenAI SDK's bearer-credential field because the subscription endpoint uses
its Responses API transport.

## Installation

```console
python -m pip install spacexai-subscription-client
```

Python 3.12 or newer is required.

## Usage

The caller owns both HTTP sessions and is responsible for closing them. Device
authorization returns a code and URL that the application should present to the
user while `async_poll_device_token` waits for approval.

```python
import aiohttp
import httpx

from spacexai_subscription_client import SpaceXAISubscriptionClient, Message


async def connect() -> None:
    async with (
        aiohttp.ClientSession() as websession,
        httpx.AsyncClient() as http_client,
    ):
        client = SpaceXAISubscriptionClient(websession, http_client)
        authorization = await client.async_request_device_authorization()
        print(authorization.verification_uri_complete)
        print(authorization.user_code)
        token = await client.async_poll_device_token(authorization)
        access_token = token.data["access_token"]
        account = await client.async_get_account(access_token)
        models = await client.async_list_models(access_token)
        if not models:
            raise RuntimeError("No Grok models are available for this account")

        response = await client.async_create_response(
            access_token,
            model=models[0],
            input_data=[Message("user", f"Hello, {account.display_name}")],
            tools=[],
        )
        print(response.text)
```

`OAuthToken.as_dict()` returns a mutable copy suitable for caller-owned
persistence. Refresh is intentionally not implemented by this package; the
host application should refresh tokens through its OAuth framework and pass the
current access token to each request.

## API boundary

The package exposes provider-neutral dataclasses for messages, tools, tool
calls, tool results, accounts, and completions. Provider SDK types do not cross
the public boundary. Network failures are translated into the stable exception
hierarchy rooted at `SpaceXAISubscriptionError`.

Model discovery and response generation use explicit timeouts. SDK retries are
disabled so callers receive a single stable failure and can apply their own
retry policy without duplicating a response request.

Video generation follows the provider's deferred start-and-poll contract with
bounded request and overall polling timeouts. Completed videos expose their
temporary HTTPS URL and metadata so the host can persist the media promptly.

The OAuth client identity and provider endpoints are centralized in
`spacexai_subscription_client.const` so an upstream identity decision can be
adopted without changing the public client API.

## Protocol provenance

The OAuth device flow, scopes, client identity, subscription proxy, and
Responses transport track the current public
[Grok Build authentication guide](https://github.com/xai-org/grok-build/blob/main/crates/codegen/xai-grok-pager/docs/user-guide/02-authentication.md)
and [Grok Build source](https://github.com/xai-org/grok-build). Keeping this
reference explicit makes upstream protocol changes reviewable without depending
on or executing the Grok CLI.

## Development

```console
uv sync --locked
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy --strict src/spacexai_subscription_client
uv run --no-sync pytest --cov=spacexai_subscription_client --cov-fail-under=95
```

Releases are built and published from the public GitHub Actions workflow using
PyPI trusted publishing. See [RELEASING.md](RELEASING.md) for the release
checklist and [CHANGELOG.md](CHANGELOG.md) for release notes.
