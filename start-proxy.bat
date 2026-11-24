@echo off
rem Start the thinking-fixup proxy daemon.
rem
rem Edit the two lines below for your own relays. Each entry is
rem <local-port>=<upstream-host>; the upstream scheme defaults to https.
rem
rem Keep one port per relay so switching providers in your client does not
rem require restarting anything.

set "OD_PROXY_ROUTES=18997=relay-a.example.com,18996=relay-b.example.com"

rem Optional tuning (defaults shown):
rem   OD_PROXY_EMPTY_RETRIES=3      retries when reasoning comes back empty
rem   OD_PROXY_RETRY_MAX_INPUT=60000 skip retries on large prompts
rem   OD_PROXY_ABORT_TEXT_MIN=20    abort once this much text streamed with no thinking
rem   OD_PROXY_DECIDE_TIMEOUT=180   stop waiting for a stalled upstream

start "" pythonw -I "%~dp0relay_think_fix.py"
