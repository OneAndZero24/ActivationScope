#!/usr/bin/env bash
# Run all matrix combos via docker compose (cpu by default).
#
# Usage:
#   scripts/run_tests.sh                   # run all combos (cpu)
#   scripts/run_tests.sh --platform cu124  # run all combos (cuda cu124)
set -euo pipefail

REPO=${REPO:-$(git rev-parse --show-toplevel)}
cd "${REPO}"

# Parse arguments to forward to the compose generator.
PLATFORM_ARGS=()
while [[ $# -gt 0 ]]; do
    PLATFORM_ARGS+=("$1")
    shift
done

# Generate pyproject.toml and docker-compose.yml.
python utils/generate-pyproject.py
python utils/generate-compose.py "${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"}"

cd .docker
docker compose build
docker compose up
cd ..

printf "\n✅ All tests passed.\n"
