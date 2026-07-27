#!/usr/bin/env bash
# Run every data-generation consistency check.
#
# Each suite compares a generator against an independent dense construction, so all
# of them diagonalize small systems exactly and take a few minutes. Run from the
# repository root:
#
#   bash tests/run_tests.sh
#
# Set PYTHON or JULIA to override the interpreters.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

PYTHON="${PYTHON:-.venv/bin/python}"
JULIA="${JULIA:-julia}"
[ -x "$PYTHON" ] || PYTHON="python3"

failed=0

run() {
    local label="$1"; shift
    echo "════════════════════════════════════════════════════════════════"
    echo "  $label"
    echo "════════════════════════════════════════════════════════════════"
    if "$@"; then
        echo "  -> PASSED: $label"
    else
        echo "  -> FAILED: $label"
        failed=$((failed + 1))
    fi
    echo
}

run "zero-field XXZ (generate_data_heisenberg.py)" \
    "$PYTHON" tests/data_generation/test_generate_data_heisenberg.py

run "XXZ in a field (generate_data_heisenberg_field.py)" \
    "$PYTHON" tests/data_generation/test_generate_data_heisenberg_field.py

run "TFIM ground states (generate_data_tfim.jl)" \
    "$JULIA" --project=. tests/data_generation/test_generate_data_tfim.jl

run "TFIM thermal IPR kernel (generate_data_tfim_thermal_ipr_kernel.jl)" \
    "$JULIA" --project=. tests/data_generation/test_generate_data_tfim_thermal_ipr_kernel.jl

run "TFIM thermal HS kernel (generate_data_tfim_thermal_hs_kernel.jl)" \
    "$JULIA" --project=. tests/data_generation/test_generate_data_tfim_thermal_hs_kernel.jl

if [ "$failed" -eq 0 ]; then
    echo "All 5 suites passed."
else
    echo "$failed suite(s) failed."
fi
exit "$failed"
