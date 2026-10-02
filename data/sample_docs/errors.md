---
title: Errors and status codes
url: https://docs.acme-tasks.example/errors
---
# Errors and status codes

Errors use a consistent JSON shape: `{"error": {"code": "...", "message": "...", "request_id": "req_..."}}`. Include the `request_id` when you contact support.

## Status codes

- **400** the request was malformed.
- **401** the key or token is missing, invalid or expired (`token_expired`).
- **403** the key does not have permission for this resource.
- **404** the resource does not exist.
- **409** a conflict, for example a duplicate idempotency key with a different body.
- **422** validation failed; the `message` names the invalid field.
- **429** rate limit exceeded.
- **500 / 503** a server problem; retry with backoff.
