#!/bin/bash
fuser -k 7890/tcp 2>/dev/null && echo "llamaloadergui stopped" || echo "was not running"
