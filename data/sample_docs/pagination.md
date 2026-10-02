---
title: Pagination
url: https://docs.acme-tasks.example/pagination
---
# Pagination

List endpoints use cursor-based pagination rather than page numbers.

## Parameters

Use `limit` to set the page size. The default is 25 and the maximum is 100. Each response contains a `next_cursor` value. Pass it as the `cursor` query parameter to fetch the next page. When `next_cursor` is `null` you have reached the last page.

## Example

```bash
curl "https://api.acme-tasks.example/v1/tasks?limit=50&cursor=eyJpZCI6MTAwfQ" \
  -H "Authorization: Bearer sk_test_123"
```

Cursors are opaque and expire after 24 hours, so do not store them long term. Results are ordered by creation time, newest first.
