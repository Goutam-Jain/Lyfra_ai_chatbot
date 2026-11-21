#!/bin/sh
# Writes the browser runtime config from environment variables at container start,
# so one image works against any backend URL.
set -e
API_URL="${API_BASE_URL:-http://localhost:8000}"
API_URL_ESCAPED=$(printf '%s' "$API_URL" | sed 's/["\\]/\\&/g')
printf 'window.__APP_CONFIG__ = { API_BASE_URL: "%s" };\n' "$API_URL_ESCAPED" > /usr/share/nginx/html/env.js
echo "runtime-env: API_BASE_URL=$API_URL"
