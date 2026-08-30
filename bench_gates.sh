#!/usr/bin/env bash
# Per-gate latency benchmark for ControlPlane Checker.
#
#   ./bench_gates.sh              # default: 30 iters, all modes
#   ./bench_gates.sh -n 100       # more iterations (tighter p99)
#   ./bench_gates.sh -m latency   # per-gate latency only
#   ./bench_gates.sh -m isolated  # one gate at a time (no CPU contention)
#   ./bench_gates.sh -m length    # how output-gate cost scales with premise length
#   ./bench_gates.sh -m conc -c 16   # concurrency ramp
#
# Requires: curl, jq, awk. Run against llm.backend = "mock".

set -uo pipefail

HOST="${HOST:-localhost:8080}"
ITERS=30
WARMUP=5
CONC=8
MODE="all"

while getopts "n:c:m:h:" opt; do
  case $opt in
    n) ITERS="$OPTARG" ;;
    c) CONC="$OPTARG" ;;
    m) MODE="$OPTARG" ;;
    h) HOST="$OPTARG" ;;
    *) echo "usage: $0 [-n iters] [-c concurrency] [-m latency|isolated|length|conc|all]" >&2; exit 1 ;;
  esac
done

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# ---------------------------------------------------------------- helpers

hr() { printf '%*s\n' 84 '' | tr ' ' '-'; }

# stats <label>  -- reads numbers on stdin
stats() {
  sort -n | awk -v label="$1" '
    function pct(p,   i) { i = int(NR * p + 0.5); if (i < 1) i = 1; if (i > NR) i = NR; return a[i] }
    { a[NR] = $1; sum += $1 }
    END {
      if (NR == 0) { printf "  %-16s  no samples\n", label; exit }
      printf "  %-16s  n=%-4d  min=%7.2f  p50=%7.2f  p90=%7.2f  p99=%7.2f  max=%7.2f  mean=%7.2f\n",
             label, NR, a[1], pct(0.50), pct(0.90), pct(0.99), a[NR], sum/NR
    }'
}

# post <payload> -- returns response body
post() {
  curl -s "http://$HOST/v1/check" -H 'content-type: application/json' -d "$1"
}

# Pull per-gate latency into $TMP/<gate>.lat and pipeline timings into $TMP/_<stage>.lat
collect() {
  jq -r '
    (.gates[]? | "GATE \(.gate) \(.latency_ms // .latency // "null")"),
    (.timings  | "STAGE prefilter \(.prefilter_ms)",
                 "STAGE input \(.input_gates_ms)",
                 "STAGE llm \(.llm_ms)",
                 "STAGE output \(.output_gates_ms)",
                 "STAGE total \(.total_ms)")
  ' 2>/dev/null | while read -r kind name val; do
    [ "$val" = "null" ] && continue
    case $kind in
      GATE)  echo "$val" >> "$TMP/$name.lat" ;;
      STAGE) echo "$val" >> "$TMP/_$name.lat" ;;
    esac
  done
}

# run_series <label> <payload>
run_series() {
  local label="$1" payload="$2" i
  rm -f "$TMP"/*.lat
  for ((i = 0; i < WARMUP; i++)); do post "$payload" >/dev/null; done
  for ((i = 0; i < ITERS; i++)); do post "$payload" | collect; done

  hr; echo "$label"; hr
  for f in "$TMP"/*.lat; do
    [ -e "$f" ] || continue
    local g; g=$(basename "$f" .lat)
    [[ "$g" == _* ]] && continue
    stats "$g" < "$f"
  done
  echo "  ---"
  for s in prefilter input llm output total; do
    [ -e "$TMP/_$s.lat" ] && stats "[$s]" < "$TMP/_$s.lat"
  done
  echo
}

# ---------------------------------------------------------------- payloads

Q_SHORT='{"query":"explain rust ownership rules","options":{"dry_run":true}}'

Q_WITH_HIST='{
  "query":"what database did we pick?",
  "history_summary":"We agreed to use PostgreSQL with connection pooling via pgbouncer.",
  "options":{"dry_run":true}
}'

# Build a history_summary of roughly N words.
hist_payload() {
  local words=$1
  python3 - "$words" <<'PY'
import json, sys
n = int(sys.argv[1])
sentence = "The team evaluated PostgreSQL and MySQL for the migration. "
hist = (sentence * ((n // 9) + 1)).strip()
print(json.dumps({
    "query": "what is the migration plan?",
    "history_summary": hist,
    "options": {"dry_run": True},
}))
PY
}

# ---------------------------------------------------------------- modes

mode_latency() {
  echo
  echo "################ PER-GATE LATENCY (all gates concurrent) ################"
  echo "This is the realistic number: gates contend for CPU as they do in production."
  echo
  run_series "short query, no history" "$Q_SHORT"
  run_series "query + history (output rail active)" "$Q_WITH_HIST"
}

mode_isolated() {
  echo
  echo "################ ISOLATED GATE LATENCY ################"
  echo "One gate at a time. Shows true single-gate cost without pool/CPU contention."
  echo "Compare against the concurrent numbers above: a large gap means you are"
  echo "core-bound, not model-bound."
  echo
  local gates=(coherence toxicity intent pii)
  for keep in "${gates[@]}"; do
    local skip=() g
    for g in "${gates[@]}" grounding relevance; do
      [ "$g" = "$keep" ] || skip+=("\"$g\"")
    done
    local list; list=$(IFS=,; echo "${skip[*]}")
    run_series "only: $keep" \
      "{\"query\":\"explain rust ownership rules\",\"options\":{\"dry_run\":true,\"skip_gates\":[$list]}}"
  done
}

mode_length() {
  echo
  echo "################ SEQUENCE-LENGTH SCALING ################"
  echo "Output-gate cost is dominated by premise length, not parameter count."
  echo "This is where your latency budget actually goes."
  echo
  for w in 10 50 150 300 600; do
    run_series "history ~${w} words" "$(hist_payload $w)"
  done
}

mode_conc() {
  echo
  echo "################ CONCURRENCY RAMP ################"
  echo "Watch p99 vs p50. If p99 blows up while p50 holds, you have a blocking"
  echo "call outside spawn_blocking, or pool_size_per_model is too small."
  echo "Check cp_inference_pool_wait_seconds in /metrics alongside this."
  echo
  for c in 1 2 4 "$CONC"; do
    rm -f "$TMP"/*.lat
    local i j
    for ((i = 0; i < WARMUP; i++)); do post "$Q_WITH_HIST" >/dev/null; done
    for ((i = 0; i < ITERS; i++)); do
      for ((j = 0; j < c; j++)); do
        post "$Q_WITH_HIST" > "$TMP/resp.$j" &
      done
      wait
      for ((j = 0; j < c; j++)); do collect < "$TMP/resp.$j"; done
    done
    hr; echo "concurrency = $c  (${ITERS} rounds x ${c} parallel)"; hr
    for f in "$TMP"/*.lat; do
      [ -e "$f" ] || continue
      local g; g=$(basename "$f" .lat)
      [[ "$g" == _* ]] && continue
      stats "$g" < "$f"
    done
    echo "  ---"
    for s in input output total; do
      [ -e "$TMP/_$s.lat" ] && stats "[$s]" < "$TMP/_$s.lat"
    done
    echo
  done
}

# ---------------------------------------------------------------- main

if ! curl -sf "http://$HOST/readyz" >/dev/null; then
  echo "ERROR: service unreachable at $HOST -- is 'cargo run' up?" >&2
  exit 1
fi

echo "Model status:"
curl -s "http://$HOST/readyz" | jq -r '.models[] | "  \(.model_id)\t\(.backend)\tlive=\(.is_live)"'

STUB=$(curl -s "http://$HOST/readyz" | jq '[.models[]|select(.is_live==false)]|length')
[ "$STUB" -ne 0 ] && echo "WARNING: $STUB model(s) stubbed -- latencies below are not real."

# A stray latency_ms of null on every gate means the field is missing from the API.
probe=$(post "$Q_SHORT" | jq -r '[.gates[]?|.latency_ms // .latency // empty]|length')
if [ "${probe:-0}" -eq 0 ]; then
  echo
  echo "ERROR: no per-gate latency field in the response."
  echo "The GateOutcome schema has 'latency' but the handler may not be serialising it."
  echo "Add latency_ms to each gate object in the /v1/check response, then re-run."
  exit 1
fi

echo "iters=$ITERS warmup=$WARMUP host=$HOST mode=$MODE"

case "$MODE" in
  latency)  mode_latency ;;
  isolated) mode_isolated ;;
  length)   mode_length ;;
  conc)     mode_conc ;;
  all)      mode_latency; mode_isolated; mode_length ;;
  *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

hr
cat <<'EOF'
Budget reference (fp32, one core):
  input rail   target < 100 ms p50
  output rail  target < 150 ms p50
Past ~200 ms p50 the guardrail becomes the product's latency story.

What to look for:
  * a gate whose p99 is >3x its p50  -> contention or a blocking call
  * isolated vs concurrent gap       -> you are core-bound; raise pool_size or quantize
  * output rail scaling steeply with premise length -> lower max_premise_tokens
EOF