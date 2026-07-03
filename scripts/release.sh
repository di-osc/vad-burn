#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release.sh v0.1.0 [options]

Trigger the GitHub Release workflow. By default it publishes the GitHub
Release, PyPI wheels, and crates.io crates from the selected ref.

Options:
  --ref REF          Git ref to run the workflow from (default: current branch)
  --no-pypi         Do not publish to PyPI
  --no-crates       Do not publish to crates.io
  --watch           Watch the GitHub Actions run until completion
  -h, --help        Show this help
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

tag="$1"
ref="$(git branch --show-current)"
publish_pypi=true
publish_crates=true
watch=false
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      ref="${2:-}"
      if [[ -z "$ref" ]]; then
        echo "--ref requires a value" >&2
        exit 2
      fi
      shift
      ;;
    --no-pypi)
      publish_pypi=false
      ;;
    --no-crates)
      publish_crates=false
      ;;
    --watch)
      watch=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ -z "$ref" ]]; then
  echo "could not determine the current branch; pass --ref REF" >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh is required: https://cli.github.com/" >&2
  exit 1
fi

gh workflow run release.yml \
  --ref "$ref" \
  --field "tag=$tag" \
  --field "publish=$publish_pypi" \
  --field "publish_crates=$publish_crates"

echo "Triggered release workflow for $tag on $ref"

if [[ "$watch" == true ]]; then
  sleep 3
  gh run watch "$(gh run list --workflow release.yml --branch "$ref" --limit 1 --json databaseId --jq '.[0].databaseId')"
fi
