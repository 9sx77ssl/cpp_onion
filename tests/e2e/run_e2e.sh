#!/usr/bin/env bash
# End-to-end: search a 1-char prefix, then verify the output with the
# independent Python oracle.
set -euo pipefail

BIN="$1"
ORACLE="$2"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# 2-char prefix: ~1024 expected tries — the Phase 0 exit gate (design doc §22)
"$BIN" ab --count 1 --threads 2 --out "$OUT" --quiet

DIR="$(find "$OUT" -mindepth 1 -maxdepth 1 -type d | head -n1)"
test -n "$DIR" || { echo "FAIL: no result directory produced"; exit 1; }

case "$(basename "$DIR")" in
  ab*) ;;
  *) echo "FAIL: result does not start with requested prefix"; exit 1 ;;
esac

python3 "$ORACLE" "$DIR"
