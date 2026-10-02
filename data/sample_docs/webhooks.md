---
title: Webhooks
url: https://docs.acme-tasks.example/webhooks
---
# Webhooks

Webhooks notify your server when something changes. Supported events include `task.created`, `task.updated`, `task.completed` and `task.deleted`.

## Verifying signatures

Each delivery has an `X-Acme-Signature` header containing an HMAC-SHA256 of the raw request body, computed with your webhook signing secret. Compute the HMAC over the raw bytes of the body, not over a parsed and re-serialized JSON object, and compare using a constant-time function.

## Delivery and retries

Your endpoint must return a 2xx status within 10 seconds. If it does not, Acme retries the delivery up to 5 times with exponential backoff over 24 hours. After that the delivery is marked failed and you can replay it from the dashboard.

## Best practices

Respond quickly and process the event asynchronously. Deduplicate events using the `event_id` field because the same event can be delivered more than once.
