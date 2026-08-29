# Changelog

## 0.4.0

- Add batch speech transcription.
- Add unary text-to-speech synthesis with language, voice, speed, and codec controls.

## 0.3.0

- Add structured response formats for AI data tasks.
- Add image generation and multi-image editing.

## 0.2.0

- Add image and PDF attachments to user messages.
- Add provider-hosted web search, X search, and code interpreter tools.

## 0.1.0

- Publish as the explicitly unofficial `spacexai-subscription-client` distribution.
- Add OAuth device authorization and token polling.
- Distinguish an expired device authorization from a request timeout.
- Add account and model discovery.
- Add normalized Responses API conversation and tool-call support.
- Require caller-owned `aiohttp` and `httpx` sessions.
- Bound provider requests with explicit timeouts and disable implicit SDK retries.
