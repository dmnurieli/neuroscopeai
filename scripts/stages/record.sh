#!/bin/bash
# Records the graduation lesson's fixed prompts on every OLMo 2 1B stage, ONE stage on disk at a time:
# download (pinned commit) -> check sizes -> record -> delete. Output: one JSON line per prompt per stage in
# $OUT (default scripts/stages/recordings.jsonl); then `python3 -I scripts/stages/to_swift.py` writes
# app/Sources/AirlineKit/GraduationRecordings.swift.
#
#   scripts/stages/record.sh [stage-id …]      (default: every stage below)
#   WORK=/some/dir  where downloads land (default: $TMPDIR/olmo-stages); LOCAL_<id>=DIR uses an existing copy
#
# Needs ~6 GB free beyond a 6 GB floor for the largest stage (the F32 base checkpoints); refuses otherwise.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$HERE/recordings.jsonl}"
WORK="${WORK:-${TMPDIR:-/tmp}/olmo-stages}"
TARGET="${CARGO_TARGET_DIR:-$ROOT/engine/target}"
BIN="$TARGET/release/stage-record"
AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# id | repo | branch | set | trained word pieces | label
STAGES='
step0|allenai/OLMo-2-0425-1B|stage1-step0-tokens0B|snapshot|0|before any training
step300|allenai/OLMo-2-0425-1B|stage1-step300-tokens1B|snapshot|1000000000|after 1 billion word pieces
step10000|allenai/OLMo-2-0425-1B|stage1-step10000-tokens21B|snapshot|21000000000|after 21 billion word pieces
step100000|allenai/OLMo-2-0425-1B|stage1-step100000-tokens210B|snapshot|210000000000|after 210 billion word pieces
step1000000|allenai/OLMo-2-0425-1B|stage1-step1000000-tokens2098B|snapshot|2098000000000|after 2.1 trillion word pieces
step1907359|allenai/OLMo-2-0425-1B|stage1-step1907359-tokens4001B|snapshot|4001000000000|after 4 trillion word pieces
base|allenai/OLMo-2-0425-1B|main|base|4001000000000|base model (finished pretraining)
sft|allenai/OLMo-2-0425-1B-SFT|main|chat|0|after flight school (SFT)
dpo|allenai/OLMo-2-0425-1B-DPO|main|chat|0|after the check ride (DPO)
rlvr1|allenai/OLMo-2-0425-1B-RLVR1|main|chat|0|after checkable rewards (RLVR, round 1)
instruct|allenai/OLMo-2-0425-1B-Instruct|main|chat|0|final model (RLVR, round 2)
'

(cd "$HERE/record" && CARGO_TARGET_DIR="$TARGET" cargo build --release --quiet)
mkdir -p "$WORK"

want=("$@")
while IFS='|' read -r id repo branch set tokens label; do
  [ -z "$id" ] && continue
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $id "* ]]; then continue; fi
  commit="$(curl -sf "https://huggingface.co/api/models/$repo/revision/$branch" | python3 -I -c 'import json,sys;print(json.load(sys.stdin)["sha"])')"
  local_var="LOCAL_$id"
  dir="${!local_var:-}"
  fetched=0
  if [ -z "$dir" ]; then
    dir="$WORK/$id"
    files="$(curl -sf "https://huggingface.co/api/models/$repo/tree/$commit" | python3 -I -c '
import json,sys
keep=("config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","model.safetensors.index.json")
for x in json.load(sys.stdin):
    p=x["path"]
    if p in keep or p.endswith(".safetensors") or p=="pytorch_model.bin": print(p, x.get("size",0))')"
    need=$(echo "$files" | awk '{s+=$2} END {print int(s/1073741824)+1}')
    avail=$(df -g "$WORK" | awk 'NR==2 {print $4}')
    if [ "$avail" -lt $((need + 6)) ]; then echo "refusing $id: needs ${need} GB, ${avail} GB free" >&2; exit 1; fi
    mkdir -p "$dir"
    echo "== $id: downloading ${need} GB of $repo@${commit:0:8} ($branch)" >&2
    while read -r f size; do
      curl -sfL --retry 3 -C - -o "$dir/$f" "https://huggingface.co/$repo/resolve/$commit/$f"
      got=$(stat -f %z "$dir/$f")
      if [ "$got" != "$size" ]; then echo "size mismatch $f: $got != $size" >&2; exit 1; fi
    done <<< "$files"
    fetched=1
  fi
  echo "== $id: recording ($set)" >&2
  "$BIN" --model "$dir" --stage "$id" --label "$label" --repo "$repo" --branch "$branch" --commit "$commit" \
         --set "$set" --tokens "$tokens" --recorded-at "$AT" --out "$OUT"
  if [ "$fetched" = 1 ]; then rm -rf "$dir"; echo "== $id: deleted $dir" >&2; fi
done <<< "$STAGES"
echo "wrote $OUT" >&2
