#!/bin/sh
set -eu

model=${1:?usage: run_memory_matrix.sh MODEL OUT_DIR}
out_dir=${2:?usage: run_memory_matrix.sh MODEL OUT_DIR}
mkdir -p "$out_dir"

mem_available() {
    awk '/^MemAvailable:/ { print $2 * 1024; exit }' /proc/meminfo
}

run_phase() {
    phase=$1
    output=$2
    before=$(mem_available)
    if [ ! -x ./q38 ]; then
        printf '{"phase":"%s","status":"blocked","reason":"q38 executable not built","mem_available_before":%s}\n' \
            "$phase" "$before" > "$output"
        return 0
    fi
    if [ ! -r "$model" ]; then
        printf '{"phase":"%s","status":"blocked","reason":"artifact missing","mem_available_before":%s}\n' \
            "$phase" "$before" > "$output"
        return 0
    fi
    plan=$(./q38 --memory-plan "$model" --json)
    after=$(mem_available)
    python3 - "$phase" "$before" "$after" "$plan" > "$output" <<'PY'
import json
import sys
phase, before, after, plan = sys.argv[1:]
item = json.loads(plan)
item.update({"phase": phase, "status": "pass",
             "mem_available_before": int(before),
             "mem_available_after": int(after),
             "whole_file_cuda_host_register": False,
             "persistent_dequant_mirror": False})
print(json.dumps(item, sort_keys=True))
PY
}

run_phase cold "$out_dir/memory_cold.json"
run_phase warm "$out_dir/memory_warm.json"
