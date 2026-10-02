---
title: Limits and data retention
url: https://docs.acme-tasks.example/limits
---
# Limits and data retention

## Object limits

A project can hold up to 10,000 tasks. Task titles are limited to 200 characters. Attachments can be at most 25 MB each, and uploading a larger file returns 413.

## Retention

Deleted tasks are soft-deleted and kept for 30 days, after which they are permanently removed and cannot be restored. Webhook delivery logs are kept for 14 days. Audit logs are kept for 1 year on the Pro plan.

## Exporting data

Use `GET /v1/exports` to request a full JSON export of a workspace. Exports are prepared asynchronously and the download link stays valid for 7 days.
