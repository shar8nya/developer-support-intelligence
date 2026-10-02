---
title: Rate limits
url: https://docs.acme-tasks.example/rate-limits
---
# Rate limits

Requests are limited per API key. The Free plan allows 100 requests per minute and the Pro plan allows 1,000 requests per minute. Bulk endpoints count each item in the batch as one request.

## Rate limit headers

Every response includes `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset` (a Unix timestamp for when the window resets).

## Handling 429 responses

When you exceed the limit the API returns HTTP 429 with a `Retry-After` header that states how many seconds to wait. Use exponential backoff with jitter, and never retry immediately in a tight loop. The official SDKs retry 429 responses automatically up to 3 times.
