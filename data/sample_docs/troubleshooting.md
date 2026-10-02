---
title: Troubleshooting
url: https://docs.acme-tasks.example/troubleshooting
---
# Troubleshooting

## I get 401 token_expired

Access tokens last 60 minutes. Refresh the token using your refresh token and retry the request. If refreshing also fails with 401, the refresh token was probably already used or is older than 30 days, and the user must sign in again.

## My webhook endpoint receives nothing

Check that the endpoint is publicly reachable over HTTPS and answers with a 2xx code within 10 seconds. Look at the delivery log in the dashboard to see the last error.

## Browser requests are blocked by CORS

The API does not allow cross-origin requests that carry secret keys. Call the API from your backend instead of from browser code.

## My requests are slow

Use pagination with a reasonable `limit`, avoid fetching every task on each page load, and reuse a single HTTP client so connections are kept alive.
