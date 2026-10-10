#!/usr/bin/env bash
# Layer-32 ICL edits WITH the chat template (official NLA pair; paper appendices "Chat template" and
# "Task switching: chat-template prompts"). AV servers on $AV_GPUS; one experiment worker per GPU,
# each holding the target model and the AR: answer redirection in two task shards and task switching.
# Results land in results/icl_edits/gemma3-12b/ (root overridable with $SOAR_RESULTS).
set -euo pipefail
PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-gemma3-12b}
AV_GPUS=${AV_GPUS:-4,5,6,7}
AV_PORTS=${AV_PORTS:-31000,31001,31002,31003}
REDIRECT_A_GPU=${REDIRECT_A_GPU:-0}
REDIRECT_B_GPU=${REDIRECT_B_GPU:-1}
SWITCH_GPU=${SWITCH_GPU:-3}
LOGS=${LOGS:-$($PYTHON -c 'from soar_nla.paths import results_dir; print(results_dir("icl_edits") / "logs")')}
mkdir -p "$LOGS"
$PYTHON -m soar_nla.icl_edits.run serve-av --config $CONFIG --gpus $AV_GPUS --ports $AV_PORTS --log-dir "$LOGS"
run() { $PYTHON -u -m soar_nla.icl_edits.run exp --config $CONFIG --av-ports $AV_PORTS --fmt chat "$@" ; }
run --gpu $REDIRECT_A_GPU --exp icl_redirect --only antonym capitalize country-capital --tag a > "$LOGS/l32_chat_icl_redirect_a.log" 2>&1 &
run --gpu $REDIRECT_B_GPU --exp icl_redirect --only english-french present-past singular-plural --tag b > "$LOGS/l32_chat_icl_redirect_b.log" 2>&1 &
run --gpu $SWITCH_GPU --exp icl_switch > "$LOGS/l32_chat_icl_switch.log" 2>&1 &
wait
$PYTHON -m soar_nla.icl_edits.analysis --config $CONFIG
