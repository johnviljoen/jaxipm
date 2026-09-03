#!/usr/bin/env bash
# After K + run-H on GPU 6: re-run J stage 2 (loader args-structure fix), then the clean ILB-on sweep.
set -u
cd "$(dirname "$0")/../.."
LOGD=tests/rebuttal/logs_par_20260827
log() { echo "[$(date +%F' '%T)] $*"; }
until grep -q 'K-AND-QUALITY DONE' "$LOGD/k_quality_waiter.log" 2>/dev/null; do sleep 120; done
log "K/quality done -> J stage 2 on GPU 6"
bash tests/rebuttal/run_J.sh 6 > "$LOGD/gpu6_J2.log" 2>&1; log "J2 exit $? ($(grep -c 'FAIL' "$LOGD/gpu6_J2.log") FAIL lines)"
echo "J2 DONE"
bash tests/rebuttal/ilb_on_rerun.sh
