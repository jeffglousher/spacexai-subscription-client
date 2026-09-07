# Changelog

## 0.1.0

- Publish as the explicitly unofficial `spacexai-subscription-client` distribution.
- Add OAuth device authorization and token polling.
- Distinguish an expired device authorization from a request timeout.
- Preserve the provider-issued device-code expiry across polling retries.
- Respect every OAuth slow-down response without reducing the server's polling interval.
- Distinguish permission failures from invalid OAuth credentials.
- Add account and model discovery.
- Add normalized Responses API conversation and tool-call support.
- Require caller-owned `aiohttp` and `httpx` sessions.
- Bound provider requests with explicit timeouts and disable implicit SDK retries.
