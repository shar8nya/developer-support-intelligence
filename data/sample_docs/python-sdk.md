---
title: Python SDK
url: https://docs.acme-tasks.example/sdk/python
---
# Python SDK

Install with `pip install acme-tasks`. The SDK requires Python 3.9 or newer.

```python
from acme_tasks import Client

client = Client(api_key="sk_test_123", timeout=30)
task = client.tasks.create(title="Write docs")
```

## Timeouts and retries

The default timeout is 30 seconds. The client retries network errors and 429/5xx responses up to 3 times with exponential backoff. Change this with the `max_retries` argument.

## Async client

Use `AsyncClient` inside asyncio code. Create one client instance and reuse it; creating a new client per request wastes connections.
