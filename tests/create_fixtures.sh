#!/bin/bash
# Generate static audio fixture files for tests.
# Requires: ffmpeg
# Run: bash tests/create_fixtures.sh

set -e
DIR="$(cd "$(dirname "$0")/fixtures" && pwd)"
mkdir -p "$DIR"

# 2-second 440 Hz tone as OGG (Opus)
ffmpeg -f lavfi -i "sine=frequency=440:duration=2" \
    -c:a libopus -b:a 32k "$DIR/sample.ogg" -y 2>/dev/null

echo "Created: $DIR/sample.ogg ($(stat -c%s "$DIR/sample.ogg" 2>/dev/null || stat -f%z "$DIR/sample.ogg") bytes)"
