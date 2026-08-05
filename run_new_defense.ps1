# =====================================================================
# New Defense Experiments for ProVFL (PowerShell)
# 5 new defenses: grad_clip, gauss_noise, dp_gauss, grad_sparse, random_proj
# Target: Adult dataset, property=sex
# =====================================================================

$dataset = "adult"
$property = "sex"
$gpu = 0

Write-Host "===== Running New Defense Experiments =====" -ForegroundColor Green

# ======================== 1. Gradient Clipping ========================
Write-Host "`n[1/5] Gradient Clipping defenses..." -ForegroundColor Cyan
$clip_norms = @(0.1, 0.5, 1.0, 2.0)
foreach ($norm in $clip_norms) {
    Write-Host "  Running grad_clip with max_norm=$norm"
    python vfl_pia_defense.py --dataset $dataset --property $property --gpu $gpu --defense grad_clip --d_para $norm
}

# ======================== 2. Gaussian Noise ========================
Write-Host "`n[2/5] Gaussian Noise defenses..." -ForegroundColor Cyan
$sigmas = @(0.001, 0.01, 0.05, 0.1)
foreach ($sigma in $sigmas) {
    Write-Host "  Running gauss_noise with sigma=$sigma"
    python vfl_pia_defense.py --dataset $dataset --property $property --gpu $gpu --defense gauss_noise --d_para $sigma
}

# ======================== 3. DP Gaussian (Clip + Noise) ========================
Write-Host "`n[3/5] DP Gaussian (Clip + Noise) defenses..." -ForegroundColor Cyan
$dp_params = @(
    @(1.0, 0.1),
    @(1.0, 0.5),
    @(0.5, 0.5),
    @(0.5, 1.0)
)
foreach ($params in $dp_params) {
    $clip = $params[0]
    $noise = $params[1]
    Write-Host "  Running dp_gauss with clip=$clip, noise_mult=$noise"
    python vfl_pia_defense.py --dataset $dataset --property $property --gpu $gpu --defense dp_gauss --d_para $clip --d_para2 $noise
}

# ======================== 4. Gradient Sparsification ========================
Write-Host "`n[4/5] Gradient Sparsification defenses..." -ForegroundColor Cyan
$keep_ratios = @(0.125, 0.25, 0.5, 0.75)
foreach ($ratio in $keep_ratios) {
    Write-Host "  Running grad_sparse with keep_ratio=$ratio"
    python vfl_pia_defense.py --dataset $dataset --property $property --gpu $gpu --defense grad_sparse --d_para $ratio
}

# ======================== 5. Random Projection ========================
Write-Host "`n[5/5] Random Projection defenses..." -ForegroundColor Cyan
$proj_dims = @(12, 8, 4, 2)
foreach ($dim in $proj_dims) {
    Write-Host "  Running random_proj with proj_dim=$dim"
    python vfl_pia_defense.py --dataset $dataset --property $property --gpu $gpu --defense random_proj --d_para $dim
}

Write-Host "`n===== All defense experiments completed =====" -ForegroundColor Green
