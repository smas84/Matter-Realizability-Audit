#!/usr/bin/env bash
# ============================================================
# run_audit.sh
# Reference runner for the Matter Realizability Audit (MRA)
#
# Usage:
#   bash run_audit.sh            # print to terminal
#   bash run_audit.sh --save     # also save to audit_output.txt
# ============================================================

set -euo pipefail

SCRIPT="flrw_grand_audit_final.py"
OUTPUT="audit_output.txt"

if [ ! -f "$SCRIPT" ]; then
  echo "Error: $SCRIPT not found in current directory."
  echo "Please run this script from the MatterRealizabilityAudit_v1.0/ directory."
  exit 1
fi

echo "=================================================="
echo "  Matter Realizability Audit (MRA) — v1.0"
echo "  Python: $(python3 --version)"
echo "  SymPy:  $(python3 -c 'import sympy; print(sympy.__version__)')"
echo "  Start:  $(date)"
echo "=================================================="
echo ""

if [[ "${1:-}" == "--save" ]]; then
  echo "Output will be saved to: $OUTPUT"
  echo ""
  python3 "$SCRIPT" 2>&1 | tee "$OUTPUT"
  echo ""
  echo "Output saved to $OUTPUT"
else
  python3 "$SCRIPT"
fi

echo ""
echo "=================================================="
echo "  Audit complete: $(date)"
echo "=================================================="
