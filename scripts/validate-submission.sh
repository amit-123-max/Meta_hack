#!/usr/bin/env bash
# ============================================================
# validate-submission.sh
# Usage: bash scripts/validate-submission.sh <BASE_URL> <PROJECT_DIR>
# ============================================================

BASE_URL="${1:-http://localhost:7860}"
PROJECT_DIR="${2:-.}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

PASS=0
FAIL=0
WARN=0

pass()  { echo -e "  ${GREEN}[PASS]${RESET} $1"; PASS=$((PASS+1)); }
fail()  { echo -e "  ${RED}[FAIL]${RESET} $1"; FAIL=$((FAIL+1)); }
warn()  { echo -e "  ${YELLOW}[WARN]${RESET} $1"; WARN=$((WARN+1)); }
info()  { echo -e "  ${CYAN}[INFO]${RESET} $1"; }
header(){ echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; echo -e "${BOLD}  $1${RESET}"; echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; }

py3() { python3 -c "$1" 2>/dev/null || echo ""; }

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║     OpenENV Submission Validator v1.0            ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo -e "  Target : ${YELLOW}${BASE_URL}${RESET}"
echo -e "  Project: ${YELLOW}${PROJECT_DIR}${RESET}"

# ─────────────────────────────────────────────────────────────
# 1. Project structure checks
# ─────────────────────────────────────────────────────────────
header "PHASE 1 — Project Structure"

REQUIRED_FILES=(
  "openenv.yaml"
  "requirements.txt"
  "Dockerfile"
  "app.py"
  "inference.py"
  "app/main.py"
  "env/traffic_env.py"
  "graders/easy_grader.py"
  "graders/medium_grader.py"
  "graders/hard_grader.py"
  "tasks/task_easy.py"
  "tasks/task_medium.py"
  "tasks/task_hard.py"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [ -f "${PROJECT_DIR}/${f}" ]; then
    pass "Found: ${f}"
  else
    fail "Missing: ${f}"
  fi
done

if [ -f "${PROJECT_DIR}/openenv.yaml" ]; then
  for key in "name:" "version:" "tasks:" "environment:" "inference:" "deployment:"; do
    if grep -q "${key}" "${PROJECT_DIR}/openenv.yaml"; then
      pass "openenv.yaml has key: ${key}"
    else
      fail "openenv.yaml missing key: ${key}"
    fi
  done
  for tid in "easy" "medium" "hard"; do
    if grep -q "id: ${tid}" "${PROJECT_DIR}/openenv.yaml"; then
      pass "openenv.yaml task id: ${tid}"
    else
      fail "openenv.yaml missing task id: ${tid}"
    fi
  done
fi

# ─────────────────────────────────────────────────────────────
# 2. Server connectivity
# ─────────────────────────────────────────────────────────────
header "PHASE 2 — Server Health"

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/" 2>/dev/null || echo "000")
if [ "${HTTP_STATUS}" = "200" ]; then
  pass "GET / → 200 OK"
else
  fail "GET / → ${HTTP_STATUS} (expected 200)"
fi

DOCS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/docs" 2>/dev/null || echo "000")
if [ "${DOCS_STATUS}" = "200" ]; then
  pass "GET /docs (Swagger UI) → 200 OK"
else
  warn "GET /docs → ${DOCS_STATUS}"
fi

OPENAPI_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/openapi.json" 2>/dev/null || echo "000")
if [ "${OPENAPI_STATUS}" = "200" ]; then
  pass "GET /openapi.json → 200 OK"
else
  warn "GET /openapi.json → ${OPENAPI_STATUS}"
fi

# ─────────────────────────────────────────────────────────────
# 3. Task validation helper
# ─────────────────────────────────────────────────────────────
EASY_SCORE="0"
MEDIUM_SCORE="0"
HARD_SCORE="0"

run_task_validation() {
  local TASK="$1"
  local STEPS="$2"
  local N_INTERS="$3"

  TASK_UPPER=$(echo "${TASK}" | tr '[:lower:]' '[:upper:]')
  header "PHASE 3 — Task: ${TASK_UPPER} (steps=${STEPS}, intersections=${N_INTERS})"

  # /reset
  RESET_RESP=$(curl -s -X POST "${BASE_URL}/reset" \
    -H "Content-Type: application/json" \
    -d "{\"task_id\": \"${TASK}\", \"seed\": 42}" 2>/dev/null || echo "{}")

  RESET_STATUS_FIELD=$(py3 "import json; d=json.loads('''${RESET_RESP//\'/}'''); print(d.get('status',''))" 2>/dev/null || echo "")
  # Use python file approach for robustness
  echo "${RESET_RESP}" > /tmp/oe_resp.json

  RESET_STATUS_FIELD=$(python3 -c "import json; d=json.load(open('/tmp/oe_resp.json')); print(d.get('status',''))" 2>/dev/null || echo "")
  if [ "${RESET_STATUS_FIELD}" = "ok" ]; then
    pass "/reset → status: ok"
  else
    fail "/reset → unexpected: $(head -c 150 /tmp/oe_resp.json)"
    return 1
  fi

  TASK_ID_FIELD=$(python3 -c "import json; d=json.load(open('/tmp/oe_resp.json')); print(d.get('task_id',''))" 2>/dev/null || echo "")
  if [ "${TASK_ID_FIELD}" = "${TASK}" ]; then
    pass "/reset → task_id: ${TASK}"
  else
    fail "/reset → task_id mismatch: got '${TASK_ID_FIELD}'"
  fi

  N_INT_FIELD=$(python3 -c "import json; d=json.load(open('/tmp/oe_resp.json')); print(d.get('n_intersections',''))" 2>/dev/null || echo "")
  if [ "${N_INT_FIELD}" = "${N_INTERS}" ]; then
    pass "/reset → n_intersections = ${N_INTERS}"
  else
    fail "/reset → n_intersections: got '${N_INT_FIELD}', expected '${N_INTERS}'"
  fi

  HAS_FRAME=$(python3 -c "import json; d=json.load(open('/tmp/oe_resp.json')); obs=d.get('observation',{}); print('yes' if 'frame_b64_png' in obs else 'no')" 2>/dev/null || echo "no")
  [ "${HAS_FRAME}" = "yes" ] && pass "/reset obs has frame_b64_png" || fail "/reset obs missing frame_b64_png"

  HAS_META=$(python3 -c "import json; d=json.load(open('/tmp/oe_resp.json')); obs=d.get('observation',{}); print('yes' if 'metadata' in obs else 'no')" 2>/dev/null || echo "no")
  [ "${HAS_META}" = "yes" ] && pass "/reset obs has metadata" || fail "/reset obs missing metadata"

  # /state
  curl -s "${BASE_URL}/state" > /tmp/oe_state.json 2>/dev/null || echo "{}" > /tmp/oe_state.json
  HAS_STEP=$(python3 -c "import json; d=json.load(open('/tmp/oe_state.json')); print('yes' if 'step' in d else 'no')" 2>/dev/null || echo "no")
  [ "${HAS_STEP}" = "yes" ] && pass "GET /state → has 'step' field" || fail "GET /state → missing 'step' field"

  HAS_INTERS=$(python3 -c "import json; d=json.load(open('/tmp/oe_state.json')); print('yes' if 'intersections' in d else 'no')" 2>/dev/null || echo "no")
  [ "${HAS_INTERS}" = "yes" ] && pass "GET /state → has 'intersections' field" || fail "GET /state → missing 'intersections' field"

  # /render
  RENDER_CT=$(curl -s -o /dev/null -w "%{content_type}" "${BASE_URL}/render" 2>/dev/null || echo "")
  if echo "${RENDER_CT}" | grep -q "image/png"; then
    pass "GET /render → content-type image/png"
  else
    fail "GET /render → unexpected content-type: ${RENDER_CT}"
  fi

  # /step loop
  ACTION=$(python3 -c "import json; print(json.dumps([0]*${N_INTERS}))" 2>/dev/null || echo "[0]")
  info "Running ${STEPS} steps..."
  STEP_FAIL=0
  DONE_AT=0
  for i in $(seq 1 "${STEPS}"); do
    curl -s -X POST "${BASE_URL}/step" \
      -H "Content-Type: application/json" \
      -d "{\"action\": ${ACTION}}" > /tmp/oe_step.json 2>/dev/null || echo "{}" > /tmp/oe_step.json

    HAS_REWARD=$(python3 -c "import json; d=json.load(open('/tmp/oe_step.json')); print('yes' if 'reward' in d else 'no')" 2>/dev/null || echo "no")
    if [ "${HAS_REWARD}" = "no" ] && [ "${STEP_FAIL}" -eq 0 ]; then
      fail "/step response missing 'reward' at step ${i}: $(cat /tmp/oe_step.json | head -c 120)"
      STEP_FAIL=1
    fi
    DONE_VAL=$(python3 -c "import json; d=json.load(open('/tmp/oe_step.json')); print(d.get('done', False))" 2>/dev/null || echo "False")
    if [ "${DONE_VAL}" = "True" ]; then
      DONE_AT=${i}
      info "Episode done at step ${i}"
      break
    fi
  done
  [ "${STEP_FAIL}" -eq 0 ] && pass "/step → ${STEPS} steps completed successfully"

  # /analytics
  ANA_HTTP=$(curl -s -o /tmp/oe_ana.json -w "%{http_code}" "${BASE_URL}/analytics" 2>/dev/null || echo "000")
  [ "${ANA_HTTP}" = "200" ] && pass "GET /analytics → 200 OK" || fail "GET /analytics → ${ANA_HTTP}"

  # /grade
  curl -s -X POST "${BASE_URL}/grade" > /tmp/oe_grade.json 2>/dev/null || echo "{}" > /tmp/oe_grade.json
  SCORE=$(python3 -c "import json; d=json.load(open('/tmp/oe_grade.json')); print(d.get('score','MISSING'))" 2>/dev/null || echo "MISSING")

  if [ "${SCORE}" = "MISSING" ]; then
    fail "POST /grade → missing 'score' field"
    SCORE="0"
  else
    VALID=$(python3 -c "s=float(${SCORE}); print('yes' if 0<=s<=1 else 'no')" 2>/dev/null || echo "no")
    if [ "${VALID}" = "yes" ]; then
      pass "POST /grade → score=${SCORE} ∈ [0, 1] ✓"
    else
      fail "POST /grade → score=${SCORE} out of range [0, 1]"
    fi
  fi

  # Export score
  case "${TASK}" in
    easy)   EASY_SCORE="${SCORE}" ;;
    medium) MEDIUM_SCORE="${SCORE}" ;;
    hard)   HARD_SCORE="${SCORE}" ;;
  esac
}

run_task_validation "easy"   20 1
run_task_validation "medium" 20 4
run_task_validation "hard"   20 4

# ─────────────────────────────────────────────────────────────
# 4. Difficulty monotonicity
# ─────────────────────────────────────────────────────────────
header "PHASE 4 — Difficulty Ordering (Easy ≥ Medium ≥ Hard)"

MONO=$(python3 -c "
easy=float('${EASY_SCORE}' or 0)
medium=float('${MEDIUM_SCORE}' or 0)
hard=float('${HARD_SCORE}' or 0)
print('PASS' if easy >= medium >= hard else 'FAIL')
" 2>/dev/null || echo "UNKNOWN")

if [ "${MONO}" = "PASS" ]; then
  pass "Monotonic: easy(${EASY_SCORE}) ≥ medium(${MEDIUM_SCORE}) ≥ hard(${HARD_SCORE})"
else
  fail "NOT monotonic: easy(${EASY_SCORE})  medium(${MEDIUM_SCORE})  hard(${HARD_SCORE})"
fi

# ─────────────────────────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────────────────────────
header "VALIDATION SUMMARY"
echo ""
echo -e "  Scores:"
echo -e "    Easy   : ${YELLOW}${EASY_SCORE}${RESET}"
echo -e "    Medium : ${YELLOW}${MEDIUM_SCORE}${RESET}"
echo -e "    Hard   : ${YELLOW}${HARD_SCORE}${RESET}"
echo ""
echo -e "  Results : ${GREEN}${PASS} passed${RESET}  ${RED}${FAIL} failed${RESET}  ${YELLOW}${WARN} warnings${RESET}"
echo ""

if [ "${FAIL}" -eq 0 ]; then
  echo -e "  ${GREEN}${BOLD}✅ Submission is VALID — ready for OpenENV evaluation!${RESET}"
  exit 0
else
  echo -e "  ${RED}${BOLD}❌ Submission has ${FAIL} failure(s) — fix before submitting.${RESET}"
  exit 1
fi
