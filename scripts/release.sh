#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release.sh v0.1.1 [options]

Create and push a release tag. Pushing v*.*.* triggers the GitHub Release
workflow, which builds wheels and publishes GitHub Release, PyPI, and crates.io.

Options:
  --ref REF          Git ref to tag (default: HEAD)
  --watch           Watch the run with gh when gh is installed
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
ref="HEAD"
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

git fetch --tags origin

if git rev-parse "$tag" >/dev/null 2>&1; then
  echo "tag $tag already exists locally"
else
  git tag "$tag" "$ref"
fi

git push origin "$tag"
echo "Pushed $tag. GitHub Actions will run the Release workflow."

if [[ "$watch" == true ]]; then
  if command -v gh >/dev/null 2>&1; then
    sleep 5
    gh run watch "$(gh run list --workflow release.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
  else
    echo "gh is not installed; watch the run at https://github.com/di-osc/vad-burn/actions/workflows/release.yml"
  fi
fi
