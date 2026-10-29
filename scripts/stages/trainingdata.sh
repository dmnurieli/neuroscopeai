#!/bin/bash
# Counts how often a few phrases from the graduation lesson appear in OLMo 2 1B's pretraining text
# (OLMo-mix-1124), using the public infini-gram index v4_olmo-mix-1124_llama (exact n-gram counts).
# Writes scripts/stages/trainingdata.json; to_swift.py folds it into the app. No network = no file = the
# lesson omits the counts (never invented).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/trainingdata.json"
INDEX="v4_olmo-mix-1124_llama"
AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
phrases=(
  "How do I bake a chocolate cake?"
  "bake a chocolate cake"
  "How do I bake"
  "safely on the runway"
  "The capital of France is Paris"
)
{
  printf '{"index":"%s","queried_at":"%s","counts":[' "$INDEX" "$AT"
  sep=""
  for q in "${phrases[@]}"; do
    body=$(python3 -I -c 'import json,sys;print(json.dumps({"index":sys.argv[1],"query_type":"count","query":sys.argv[2]}))' "$INDEX" "$q")
    n=$(curl -sf -m 60 -X POST https://api.infini-gram.io/ -H 'Content-Type: application/json' -d "$body" |
        python3 -I -c 'import json,sys;d=json.load(sys.stdin);assert not d.get("approx");print(d["count"])')
    printf '%s{"phrase":%s,"count":%s}' "$sep" "$(python3 -I -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$q")" "$n"
    sep=","
  done
  printf ']}\n'
} > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
cat "$OUT"
