# spacexai-client

Small async client boundary for SpaceXAI OAuth device login and the Grok
subscription API. It is designed for callers that own token persistence and
refresh, such as Home Assistant.

Authentication is OAuth-only. The client does not accept an API key and has no
API-key fallback. Internally, an OAuth access token is supplied to the OpenAI
SDK's bearer-credential field because the subscription endpoint uses its
Responses API transport.
