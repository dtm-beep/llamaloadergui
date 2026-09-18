#!/bin/bash
# Path-independent on purpose: works from any checkout location.
cd "$(dirname "$0")"
exec .venv/bin/python server.py
