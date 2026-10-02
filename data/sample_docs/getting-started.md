---
title: Getting started with the Acme Tasks API
url: https://docs.acme-tasks.example/getting-started
---
# Getting started with the Acme Tasks API

Acme Tasks is a fictional task-management service used as sample data for this project. The REST API lives at `https://api.acme-tasks.example/v1` and returns JSON.

## Create an API key

Open **Settings → Developers → API keys** in the dashboard and click **Create key**. Copy the key immediately because it is shown only once. Live keys start with `sk_live_` and test keys start with `sk_test_`.

## Make your first request

Send the key in the `Authorization` header:

```bash
curl https://api.acme-tasks.example/v1/tasks \
  -H "Authorization: Bearer sk_test_123"
```

A successful call returns HTTP 200 with a `data` array of tasks and a `next_cursor` field.

## Install an SDK

Official SDKs exist for Python and JavaScript. Install the Python SDK with `pip install acme-tasks` and the JavaScript SDK with `npm install @acme/tasks`.
