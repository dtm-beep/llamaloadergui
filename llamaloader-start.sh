#!/bin/bash
cd "$(dirname "$0")"
fuser -k 7890/tcp 2>/dev/null
sleep 0.5
.venv/bin/python server.py
