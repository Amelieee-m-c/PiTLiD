
# 10-run fixed-split protocol for grape and peach (PiTLiD's generalization datasets),
# using the same official-code architecture + fixed-split protocol that got apple's
# reproduction closest to the paper (99.20+-0.30% vs paper's 99.45+-0.17%).
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\src"

Write-Host "=== grape 10-run (fixed split, step_decay) ==="
python run_multi_seed.py --crop grape --n_runs 10 --start_seed 1 `
    --fixed_split_seed 1 `
    --output_root "../runs/grape_10run_fixedsplit_stepdecay" `
    --extra_train_args "--lr_strategy step_decay"

Write-Host "=== peach 10-run (fixed split, step_decay) ==="
python run_multi_seed.py --crop peach --n_runs 10 --start_seed 1 `
    --fixed_split_seed 1 `
    --output_root "../runs/peach_10run_fixedsplit_stepdecay" `
    --extra_train_args "--lr_strategy step_decay"

Write-Host "=== DONE: grape + peach 10-run finished ==="
