#!/usr/bin/env bash
# Baut die ball_detector Custom-Parser-.so im DeepStream-Container
# und kopiert sie nach models/. Nutzung: ./scripts/build_custom_parser.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PARSER_DIR="$PROJECT_DIR/cpp/custom_parser"
LIBS_DIR="$PROJECT_DIR/libs"
DEEPSTREAM_IMAGE="${DEEPSTREAM_IMAGE:-my-deepstream8:custom}"

if [ ! -f "$PARSER_DIR/nvdsinfer_custom_ball_parser.cpp" ]; then
  echo "Parser-Quelle nicht gefunden: $PARSER_DIR/nvdsinfer_custom_ball_parser.cpp"
  exit 1
fi

mkdir -p "$LIBS_DIR"

echo "Baue Custom-Parser im Container und schiebe ihn nach /app/libs"
sudo docker run --rm \
  -v "$PROJECT_DIR:/app" \
  -w /app/cpp/custom_parser \
  "$DEEPSTREAM_IMAGE" \
  bash -c 'make clean 2>/dev/null; make && install -m 755 libnvdsinfer_custom_ball_parser.so /app/libs/'

echo "Fertig: $LIBS_DIR/libnvdsinfer_custom_ball_parser.so"
ls -la "$LIBS_DIR/libnvdsinfer_custom_ball_parser.so" 2>/dev/null || true
