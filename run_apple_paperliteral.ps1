
# Apple 10-run under the paper's literally-described protocol (Section 3.2.2/3.2.6/3.2.7/3.2.8):
# single-stage (all layers trainable from epoch 0, None_frozen), 50 epochs with
# steps_per_epoch=200 throughout, true CLR (base=0.001/max=0.006/step_size=2000),
# real early stopping patience=14 on val_loss, L2 weight_decay=1e-3.
# Uses the fixed-split protocol (best-evidenced split reading from the 2026-09-15 test),
# orthogonal to this single-stage/CLR/early-stopping test.
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\src"

python run_multi_seed.py --crop apple --n_runs 10 --start_seed 1 `
    --fixed_split_seed 1 `
    --output_root "../runs/apple_10run_paperliteral_singlestage" `
    --extra_train_args "--single_stage --early_stopping_patience 14 --lr_strategy clr --weight_decay 1e-3"

Write-Host "=== DONE: apple paper-literal single-stage 10-run finished ==="
