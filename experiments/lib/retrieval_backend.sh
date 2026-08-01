#!/usr/bin/env bash

configure_retrieval_backend() {
   case "${RETRIEVAL_MODE}" in
      dense|lexical|hybrid) ;;
      *) echo "Unsupported --retrieval-mode ${RETRIEVAL_MODE}; expected dense, lexical, or hybrid." >&2; return 2 ;;
   esac
   case "${RETRIEVAL_BACKEND}" in
      local|serper) ;;
      *) echo "Unsupported --retrieval-backend ${RETRIEVAL_BACKEND}; expected local or serper." >&2; return 2 ;;
   esac
   if ! [[ "${SERPER_SERVER_PORT}" =~ ^[1-9][0-9]*$ ]] || [ "${SERPER_SERVER_PORT}" -gt 65535 ]; then
      echo "SERPER_SERVER_PORT must be an integer in [1, 65535]." >&2
      return 2
   fi
   if [ "${RETRIEVAL_CACHE_SIZE}" -lt 0 ] || [ "${RETRIEVAL_CONCURRENCY}" -lt 1 ]; then
      echo "Retrieval cache size must be >= 0 and concurrency must be >= 1." >&2
      return 2
   fi

   if [ -n "${RETRIEVAL_SERVER_URL+x}" ]; then
      RETRIEVAL_URL_EXPLICIT=true
   else
      RETRIEVAL_URL_EXPLICIT=false
      if [ "${RETRIEVAL_BACKEND}" = "serper" ]; then
         RETRIEVAL_SERVER_URL="http://${SERPER_SERVER_HOST}:${SERPER_SERVER_PORT}"
      else
         RETRIEVAL_SERVER_URL="http://10.2.152.50:65432"
      fi
   fi

   if [ "${RETRIEVAL_BACKEND}" = "serper" ] && [ "${RETRIEVAL_URL_EXPLICIT}" = "false" ] && [ -z "${SERPER_PROXY_TOKEN:-}" ]; then
      echo "SERPER_PROXY_TOKEN is required for the managed Serper retrieval service." >&2
      return 2
   fi

   export SERPER_SEARCH_URL="${SERPER_SEARCH_URL:-http://10.2.152.50:9999/search}"
   export RETRIEVAL_SERVER_URL
   export RLLM_RETRIEVAL_MODE="${RETRIEVAL_MODE}"
   export RLLM_RETRIEVAL_CONCURRENCY="${RETRIEVAL_CONCURRENCY}"
   export RLLM_RETRIEVAL_CACHE_SIZE="${RETRIEVAL_CACHE_SIZE}"
   if [ "${RETRIEVAL_BACKEND}" = "serper" ]; then
      export RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-10}"
   else
      export RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-${RLLM_RETRIEVAL_MAX_RESULTS:-4}}"
   fi
}

cleanup_managed_retrieval_backend() {
   local exit_status=$?
   if [ -n "${SERPER_SERVICE_PID:-}" ]; then
      kill "${SERPER_SERVICE_PID}" 2>/dev/null || true
      wait "${SERPER_SERVICE_PID}" 2>/dev/null || true
   fi
   return "${exit_status}"
}

start_managed_retrieval_backend() {
   if [ "${RETRIEVAL_BACKEND}" != "serper" ] || [ "${RETRIEVAL_URL_EXPLICIT}" = "true" ]; then
      return
   fi
   if [ "${RAY_JOB_WAIT:-0}" != "1" ] && [ "${RAY_JOB_FOLLOW_LOGS:-1}" != "1" ]; then
      echo "Managed Serper requires RAY_JOB_WAIT=1 or RAY_JOB_FOLLOW_LOGS=1, or an explicit RETRIEVAL_SERVER_URL." >&2
      return 2
   fi

   trap cleanup_managed_retrieval_backend EXIT
   python3 "${REPO_ROOT}/examples/search-r1/serper_search_server.py" \
      --host "${SERPER_SERVER_HOST}" \
      --port "${SERPER_SERVER_PORT}" \
      >"${LOG_ROOT}/serper_search_server.log" 2>&1 &
   SERPER_SERVICE_PID=$!
   python3 - "${RETRIEVAL_SERVER_URL}" "${SERPER_SERVICE_PID}" <<'PY'
import json
import os
import sys
import time
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
pid = int(sys.argv[2])
for _ in range(50):
    if not os.path.exists(f"/proc/{pid}"):
        break
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if json.load(response).get("status") == "ok":
                raise SystemExit(0)
    except Exception:
        time.sleep(0.1)
raise SystemExit(f"Managed Serper service failed to start at {url}")
PY
}
