---
title: Authentication and tokens
url: https://docs.acme-tasks.example/authentication
---
# Authentication and tokens

The API supports two authentication methods: API keys for server-to-server calls and OAuth 2.0 for apps acting on behalf of a user.

## API keys

Send the key as a bearer token: `Authorization: Bearer sk_live_...`. Keys do not expire, but you can revoke them at any time in the dashboard. Never embed a secret key in browser or mobile code.

## OAuth 2.0 access tokens

Access tokens expire after 60 minutes. When a token expires the API returns 401 with the error code `token_expired`.

## Refresh tokens

Refresh tokens are valid for 30 days. Every time you exchange a refresh token for a new access token, the response contains a new refresh token and the old one stops working (refresh token rotation). Always store the newest refresh token. Reusing an old refresh token revokes the entire token family.

## Revoking tokens

Call `POST /v1/auth/revoke` with the token you want to invalidate. Revoking a refresh token also invalidates the access tokens issued from it.
