#!/usr/bin/env bash
# Run the full Phase 0 pipeline in order.
# Stops immediately if any step fails — do not proceed to the next step
# on a failure, review the error first.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 0, Step 1: Fetch data from source repo ==="
python3 "$SCRIPT_DIR/01_fetch_data.py"

echo ""
echo "=== Phase 0, Step 2: Build labeled dataset ==="
python3 "$SCRIPT_DIR/02_build_labels.py"

echo ""
echo "=== Phase 0, Step 3: Flight-level train/val/test split ==="
python3 "$SCRIPT_DIR/03_split_flights.py"

echo ""
echo "=== Phase 0 complete ==="
echo "Review data/processed/phase0_validation_report.md before proceeding to Phase 1."
