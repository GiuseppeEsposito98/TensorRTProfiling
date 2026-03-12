#!/usr/bin/env bash

ROOT_DIR="${1:-}"

if [[ -z "$ROOT_DIR" ]]; then
  echo "Uso: $0 <root_folder>"
  exit 1
fi

ROOT_DIR="$(realpath "$ROOT_DIR")"

echo "Root path: $ROOT_DIR"

count=0
while IFS= read -r -d '' plan_path; do
  echo $plan_path
  ((count++)) || true
  plan_dir="$(dirname "$plan_path")"
  out_json="${plan_path}.json"

  echo "Profiling: $plan_path"

  # Esegui trtexec per esportare le info layer
  trtexec \
    --loadEngine="$plan_path" \
    --exportLayerInfo="$out_json" \
    --profilingVerbosity=detailed

done < <(find "$ROOT_DIR" -type f -name '*.plan' -print0)