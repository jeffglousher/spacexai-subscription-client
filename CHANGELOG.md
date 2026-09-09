# Changelog

## 0.1.1

- Separate the tested Grok Build proxy compatibility version from this package's
  version, fixing HTTP 426 responses caused by sending the package version.
- Keep the unofficial client identifier and truthful package User-Agent.
- Explicitly disable Responses API storage, matching the Grok Build sampler's
  default.

## 0.1.0

- Publish as the explicitly unofficial `spacexai-subscription-client` distribution.
- Add OAuth device authorization and token polling.
- Distinguish an expired device authorization from a request timeout.
- Preserve the provider-issued device-code expiry across polling retries.
- Respect every OAuth slow-down response without reducing the server's polling interval.
- Preserve polling backoff across retries and increase the interval after timeouts.
- Distinguish permission failures from invalid OAuth credentials.
- Add account and model discovery.
- Add normalized Responses API conversation and tool-call support.
- Require caller-owned `aiohttp` and `httpx` sessions.
- Bound provider requests with explicit timeouts and disable implicit SDK retries.
