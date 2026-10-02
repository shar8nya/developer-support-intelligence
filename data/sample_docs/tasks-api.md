---
title: Tasks endpoints and idempotency
url: https://docs.acme-tasks.example/tasks
---
# Tasks endpoints and idempotency

## Create a task

`POST /v1/tasks` with a JSON body containing `title` (required, up to 200 characters) and optional `due_date`, `assignee_id`, `labels` and `project_id`.

## Update and delete

`PATCH /v1/tasks/{id}` updates fields. `DELETE /v1/tasks/{id}` soft-deletes the task: it is hidden from list results but can be restored for 30 days before permanent removal.

## Idempotency keys

To retry a POST safely, send an `Idempotency-Key` header with a unique value such as a UUID. If the same key is sent again within 24 hours the API returns the original response instead of creating a duplicate task. Reusing a key with a different request body returns 409.
