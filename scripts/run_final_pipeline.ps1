$ErrorActionPreference = 'Stop'

function Invoke-Python {
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw 'Python stage failed; see the output above.'
    }
}

Write-Host '[1/8] Unit tests'
Invoke-Python -m pytest -q

Write-Host '[2/8] Prepare 200 new examples per benchmark (964 viewed IDs excluded)'
Invoke-Python scripts/prepare_next_final.py

Write-Host '[3/8] Verify the final holdout against its exclusion ledger'
Invoke-Python scripts/check_data_splits.py --holdout-root data_final

Write-Host '[4/8] Main comparison (Dense, Random x3, five deterministic compressed methods)'
Invoke-Python scripts/run_matrix.py --config configs/experiments/main.yaml --load-in-4bit

Write-Host '[5/8] Validate exact main STRIDE references for both budgets'
Invoke-Python scripts/run_full_ablations.py --config configs/experiments/ablations.yaml --validate-only

Write-Host '[6/8] Same-data structural ablations at 64 and 128 tokens'
Invoke-Python scripts/run_full_ablations.py --config configs/experiments/ablations.yaml --load-in-4bit

Write-Host '[7/8] Recompute every saved score'
Invoke-Python scripts/audit_scores.py --root results/main
Invoke-Python scripts/audit_scores.py --root results/ablations

Write-Host '[8/8] Paper-ready tables, statistics, plots, and journal tables'
Invoke-Python scripts/build_report.py --root results/main --output results/report --data-root data_final --ablation-root results/ablations

Write-Host 'FULL FINAL EXPERIMENT COMPLETE'
Write-Host 'Read results/report/results_report.md'
