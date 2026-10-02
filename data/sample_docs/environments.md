---
title: Sandbox and production environments
url: https://docs.acme-tasks.example/environments
---
# Sandbox and production environments

Use the sandbox for development. Its base URL is `https://sandbox.api.acme-tasks.example/v1` and it accepts only test keys that start with `sk_test_`. Data in the sandbox is reset every Sunday and never triggers real notifications or emails.

Production uses `https://api.acme-tasks.example/v1` and live keys that start with `sk_live_`. A test key used against production, or a live key used against the sandbox, returns 401 with the code `wrong_environment`.
