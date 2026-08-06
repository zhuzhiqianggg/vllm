#!/usr/bin/env bash
# 跟踪日志
cd "$(dirname "$0")/.."
docker compose logs -f --tail=200 "$@"
