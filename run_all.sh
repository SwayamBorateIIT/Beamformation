#!/bin/bash
# Train + evaluate LC-PIB and its ablations (sequential, CPU).
set -e
S=3000
run() { name=$1; shift; python3 -m lcpib.train --steps $S --out results/$name "$@" > results/train_$name.log 2>&1; python3 -m lcpib.evaluate --run results/$name > results/eval_$name.log 2>&1; echo "finished $name"; }
run full
run abl_no_ch --no_ch
run abl_xv_mse --xv_mse
run abl_xv_nocorr --xv_nocorr
run abl_no_apod --no_apod
run abl_coh_only --coh_only
run abl_no_xv --no_xv
