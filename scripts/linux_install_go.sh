#!/usr/bin/env bash
# Build and install this independent backend into a local Go Welvet checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_GO_ROOT="${WELVET_GO_ROOT:-$ROOT/../welvet}"
if [[ ! -f "$WELVET_GO_ROOT/go.mod" ]]; then
  echo "Go Welvet checkout not found: $WELVET_GO_ROOT" >&2
  exit 1
fi
"$ROOT/scripts/build_linux.sh"
DEST="$WELVET_GO_ROOT/native/rocm/linux_x86_64"
mkdir -p "$DEST" "$WELVET_GO_ROOT/native/rocm/include"
cp "$ROOT/build/libwelvet_rocm.so" "$DEST/libwelvet_rocm.so"
cp "$ROOT/include/welvet_c.h" "$WELVET_GO_ROOT/native/rocm/include/welvet_rocm.h"
VERSION="$("$ROOT/build/welvet_dense_fwd" --version)"
{
  echo "engine=welvet.rocm"
  echo "version=$VERSION"
  echo "model_abi=1"
  echo "segment_extension=1"
  echo "platform=linux_x86_64"
  echo "source_checkout=$ROOT"
  sha256sum "$DEST/libwelvet_rocm.so"
} > "$WELVET_GO_ROOT/native/rocm/MANIFEST.txt"
echo "Installed $DEST/libwelvet_rocm.so"
echo "Go build: go build -tags welvet_rocm ./..."
