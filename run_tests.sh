#!/bin/bash
#=============================================================================
# RUN TESTS — SESI Integration Test Suite
#=============================================================================

set -e

LAYER=${1:-"all"}
VERBOSE=${2:-""}

echo "=============================================="
echo " SESI Integration Test Suite"
echo "=============================================="
echo " Layer: $LAYER"
echo " Verbose: $VERBOSE"
echo "=============================================="

if [ "$LAYER" == "all" ]; then
    python integration_test_suite.py --all $VERBOSE
else
    python integration_test_suite.py --layer $LAYER $VERBOSE
fi

echo "=============================================="
echo " Tests complete!"
echo "=============================================="
