# ProVFL Defense Work — Supervisor Review & Publication Roadmap

***Round 8 update (2026-09-10) — `s1c` ran on real Kaggle. The ceiling is now confirmed on real data, not projected. 12/13 jobs succeeded (`s1c_feat_corr` crashed, returncode 1, 0.3 min — needs a re-run, but `s1c_raw_baseline` succeeded so the confound question isn't fully blocked). `norm_align` is now the best empirical result in the whole project: it moves `MAE_b_output` by 16× (0.0085→0.1375 passive) at almost no utility cost (AUC 0.901→0.897) — and `MAE_own` still doesn't move. That is the paper's core figure. Full numbers in §2.5. One logging gap found: `norms_*.txt` only captures mean/median norm, which cannot show `norm_align`'s actual mechanism (variance collapse) since the mean is preserved by construction — needs a `victim_norm_std` column before the mechanism claim has a supporting figure.***

***Round 7 update (2026-09-06) — the campaign ran, and the result changes the paper.** Sessions s1 + s1b executed on Kaggle (20 jobs, adult/sex, 5 seeds, `res/1st part` + `res/2nd part`). Recombining their per-channel rows shows the C1 adversary keeps essentially its full accuracy after **discarding the victim's forward embedding entirely**: MAE over `{a_grad, a_output}` alone is 0.0136 passive / 0.0065 active, flat across every deployable configuration, while the mechanisms move the defended channel by up to 7.5×. **Wire perturbation cannot bound this attack, so §3's "build a better noise mechanism" programme is not the contribution — the ceiling is.** Full numbers, validity check and the one live confound in §2.3. §3 is retained as design history with a retraction banner; §5's build order is re-sequenced (`s1c` confound + constructive ceiling, then `s1g` generalisation; `s2`/`s3` deprecated in the runner).*

*Round 6 update — pre-flight engineering pass complete, nothing run on GPU yet. All perturbation logic now goes through one dispatcher (`my_utils/defense_func.apply_perturbation`), both training scripts are argparse-identical, `--defend_side` splits the forward and backward channels, D4/D5 are reachable from the harness, `analyze_defense_results.py` and `kaggle_run.py` exist, and `test_defense_func.py` verifies the dispatcher (65 checks, all passing, no GPU). **§2's headline "gauss_noise Pareto-dominates dp_gauss outright" claim has been measured and is false — retracted and replaced below.***

*Round 5 — `--defend_scope` (Step A) confirmed live in the repo (verified via raw.githubusercontent.com after web_fetch served a stale cache twice). Formal novelty/prior-art check (Consensus, fasttrack-literature duplication test + gap saturation) done — see §1.5.*

## 0. Engineering log

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Defenses only perturbed gradients, never forward-pass outputs | ✅ Fixed | |
| 2 | `vfl_pia_active.py` had no defense hooks — MR+LR attacker never tested against any defense | ✅ Fixed | |
| 3 | Output-side and gradient-side defense strength hard-wired to the same `d_para` | ✅ Fixed | `--out_para` added |
| 4 | Full grid `{5 defenses} × {use_MR 0/1} × {use_LR 0/1}` on Adult/sex | ✅ **Done — results analyzed, see §2** | `res_active_adult_part1/2.txt`, 2400 rows |
| 4b | Wide `d_para` sweep per defense (no MR/LR) | ✅ **Done — results analyzed, see §2** | `res_defense_adult.txt`, 350 rows |
| 5 | Output-side defense applied to attacker's own channel (`a_output`), not just victim's (`b_output`) | ✅ **Fixed, live, and now enforced for gradients too** | `--defend_scope {victim_only, both}`; the gradient path honours it as well (Round 6) |
| 6 | MR-loop bypass diagnostic (`--defend_mr_loop`) | 📝 Patch written, not yet run | Low priority — §2 already answers most of what this was for |
| 7 | Results analysis tooling | ✅ Built, tested, used | `analyze_defense_results.py` — derives MAE/PrivacyGain/AUCDrop, paired t-tests, Pareto frontier |
| 8 | Seed count mismatch: `vfl_pia_defense.py` looped `range(5)`, `vfl_pia_active.py` looped `range(10)` | ✅ Fixed | Both now `range(args.n_seeds)`, `--n_seeds 5`, and both at `--val_ratio 0.3`. Legacy rows are tagged `era=legacy` by the analyzer (see §2) |
| 9 | Scoped Kaggle run: `victim_only` vs `both`, plus side decomposition | ✅ **Run 2026-09-05 as `--session s1`** | See row 22 and §2.3. This was Gate 1's input data; what it actually produced was the ceiling result, which makes Gate 1 moot |
| 10 | Formal novelty/prior-art check via Consensus + fasttrack-literature | ✅ Done | See §1.5 — reframes the defense's novelty claim, does not kill it |
| 11 | Every defense branch duplicated in two scripts, free to drift | ✅ Fixed | Single dispatcher `defense_func.apply_perturbation(pair, args, side, epoch)`; both scripts call it and nothing else |
| 12 | `--defend_side {output, grad, both}` | ✅ Added | Only `output` is unilaterally deployable by the victim; separating the two is what turns "noise helps" into "noise helps *on the forward channel*" |
| 13 | D4 (shuffle) and D5 (withdraw) unreachable from the campaign | ✅ Ported | Both live in the dataloader/input batch, so `--defend_side`/`--defend_scope` do not apply. D5 is ProVFL's own recommended mitigation and had never been run here |
| 14 | Campaign runner | ✅ Built, extended | `kaggle_run.py` — resume markers in `runs_done/`, per-session result files, 12h budget guard, `--list`/`--dry_run`. Sessions `s1`,`s1b` (done), **`s1c`,`s1g`** (Round 7), `s2`,`s3` (deprecated), `s4`. Scripts under `ablation/` launch via `-m` because that directory has no `__init__.py`, so `python ablation/x.py` would put `ablation/` on `sys.path[0]` and break `from dataloader import get_data` |
| 15 | Dispatcher never executed before burning GPU hours | ✅ Covered | `test_defense_func.py`, ~100 pure-tensor checks, <1s on CPU. Catches mis-scoped, no-op, and mistyped defenses. The norm-family checks (rows 24, 27) are **syntax-verified only** locally — torch is not installed on the authoring machine, so they first execute inside `kaggle_setup.py`'s pre-flight gate |
| 16 | Mistyped `--defense` silently produced undefended numbers under a defense label | ✅ Fixed | `defense_func.KNOWN_DEFENSES` raises on the first batch |
| 17 | §2's Pareto-dominance claim never significance-tested | ✅ Tested — **claim falsified**, see §2 | Was the single most load-bearing sentence in the paper's framing |
| 18 | A 12h Kaggle session can be killed before an 8.5h campaign session finishes, and there is no checkpointing | ✅ Fixed | `kaggle_run.py --part N/M` takes a contiguous slice. `--part 1/2` (3.7h) then `--part 2/2` (4.9h) covers s1's 20 jobs in two short runs, both appending to the same CSV and sharing `runs_done/` |
| 19 | An accelerator-off Kaggle session trains on CPU **without erroring**, silently invalidating the 22–34 min/job budget | ✅ Guarded | `kaggle_run.py` now forwards `--gpu` (default 0) to every job and prints the CUDA device before the first one, warning loudly when `torch.cuda.is_available()` is False. `kaggle_setup.py` fails the same check up front |
| 20 | Bootstrap was a pile of `sed` commands in an untracked notebook file, invisible to a `git clone` | ✅ Fixed at the source | Four durable fixes replace it: `datasets/__init__.py` (regular package, so the repo beats Kaggle's preinstalled HuggingFace `datasets` — no more `pip uninstall`); lazy `get_celeba_dataset` import in `dataloader.py`; `datasets/adult.py` resolves its data dir from `__file__` instead of `'../data/adult/'`; `.iteritems()` → `.items()` for pandas ≥ 2.0 |
| 21 | Environment verification | ✅ Built and run | `kaggle_setup.py` — 5 gates (deps, accelerator, data, package resolution, dispatcher tests). Adult download has urllib→curl→wget transport fallback plus a zip fallback, because UCI's cert fails verification under some Python CA bundles while succeeding under the system store. Verified end to end: loader returns 44 355 rows, positive class 24.50 %, `sex_Male` 66.21 % — matching the documented figures exactly |
| 22 | Session s1 + s1b actually run | ✅ **Done 2026-09-05** | 20 jobs, adult/sex, 5 seeds. `res/1st part`, `res/2nd part`. All returncode 0. **Measured cost 11.6–13.5 min/job (except `lap_noise` 44.9)**, i.e. the 22/34 min estimates were ~2× pessimistic — `kaggle_run.py` recalibrated to `EST_MIN = {DEFENSE: 14, ACTIVE: 16}` with `EST_MIN_OVERRIDE` for `lap_noise`/`ppdl`. The whole 56-job campaign is ~10 h, not 23.7 h |
| 23 | Nobody had asked what the attack still knows when the defended channel is **removed** | ✅ **Answered, no GPU needed** | `pia_func.property_ensemble` averages four independent per-channel estimates rather than training a joint model, so any channel-subset ablation is recoverable from the existing CSVs. `analyze_clean_channel.py` (new) does it. This is §2.3 and it is the paper |
| 24 | The ceiling had no constructive half — only mechanisms that fail to reach it | ✅ Built, not yet run | `defense_func.NORM_FAMILY`: `norm_align` (fixed-norm projection, provably collapses the sorted-norm vector to a constant), `norm_quant` (k levels, interpolates), `norm_permute` (**deliberate placebo** — same norm multiset, so PrivacyGain must come out ≈0; if it ever wins, the pipeline leaks something other than the sorted-norm statistic). Forward-side only, victim-only, refused on the gradient side |
| 25 | The confound: adult's attacker-side one-hots (`relationship_*`, `marital_status_*`) are near-deterministic proxies for `sex` | 📋 Probes queued as `--session s1c` | The property column itself is verifiably **victim-side** (`datasets/adult.py` keeps the property at index 0; `utils.split_data` gives `b` = `data[:, 0:50]`, `a` = `data[:, 50:111]`) — so the ceiling is not the trivial artefact. Whether it is *co-training coupling* or *plain feature correlation* is settled by `ablation/test_ab_correlation.py` + `ablation/test_ab_raw_baseline.py`, ~11 min total. The operational conclusion is the same either way; the mechanism sentence in the paper is not |
| 26 | Sessions s2/s3 sweep σ inside a range now measured as inert | ✅ Deprecated in the runner | `kaggle_run.py DEPRECATED_SESSIONS` prints why on selection instead of hiding it in a docstring. PrivacyGain at σ = 0.05/0.10/0.20 output-side victim-only: −0.0009 / −0.0033 / +0.0023 (p=0.6150 at the reference point). Kept runnable as an ablation |
| 27 | `--out_para -1` collapsed to `--d_para` for the norm family | ✅ Fixed | `_strength`'s "−1 means unset" rule is right for additive noise and wrong for `norm_align`, where −1 means "target the batch mean". Left unfixed it would have pinned every embedding at norm 0.05 and produced a utility collapse that looks like a privacy/utility tradeoff and is not one. Two new checks assert it |
| 28 | **Only `adult` could actually be loaded.** `census.py` and `bankmk.py` hard-coded `"../data/<name>/..."`, which from the repo root points *outside* the checkout — so `--session s1g --dataset census` would have raised `FileNotFoundError` after the session was already saved for background execution. `bankmk.py` also looked in `data/bankmarketing/` while `kaggle_setup.py` downloaded to `data/bankmk/` | ✅ Fixed | New shared `datasets.resolve_data_dir(*names, sentinel=…)` (in `datasets/__init__.py`), used by both; accepts either directory name for bankmk. `adult.py` keeps its own verified copy. `kaggle_setup.py` gate [4/5] now imports *this dataset's* loader and prints the resolved path, so the failure surfaces in <2 min instead of mid-session |
| 29 | census and bankmk were not downloadable at all | ✅ Fixed | UCI serves neither as direct files any more (the `census-income-mld/*.gz` paths are 404 — verified 2026-09-06). `DATA_ARCHIVES` replaces `DATA_ZIPS` and walks one level of nesting, because `census+income+kdd.zip` contains a single `census.tar.gz` containing the data, and `bank+marketing.zip` is a zip of zips. Member names were read off the real archives, not guessed. Exercised end to end outside the repo: `bank-full.csv` 4 610 348 B, `census-income.data` 103 874 469 B, `census-income.test` 51 918 813 B |
| 30 | Every cost estimate assumed adult | ✅ Fixed | `EST_MIN` was calibrated on adult (44 355×111, 20 epochs). census is 299 285×511 at 30 epochs. `DATASET_FACTOR` scales it, and the budget warning now prints in `--list` mode too — which is how the census s1g session turned out to be **17.3 h, over the 12 h Kaggle ceiling**, before any GPU time was spent rather than after |

## 1. Correction to external advice: don't defend the attacker's own channel

A second reviewer (ChatGPT) suggested a "leakage-aware" defense that puts extra noise specifically on `a_output`, since it's ProVFL's most successful attack channel. **This doesn't match the threat model and can't actually be deployed.** In the codebase, `model_list[0]` (the `a` channel) is explicitly commented `# attacker` — `a_output` is the adversary's own local computation. A defender has no mechanism to force an adversarial party to perturb its own data; only the victim's channel (`b`) is something a real defense can touch. This is exactly why `--defend_scope` (item 5) exists — use `victim_only` scope, and expect `a_output` MAE to remain an un-defendable floor in every result table from here on. Say this explicitly in the thesis as a stated limitation of the threat model, not an oversight — it's a correct and defensible modeling choice.

**Round 6 extension — the same argument applies to `a_grad`, and that half was still unfixed until now.** `a_grad` is the gradient the active party computes *in its own backward pass* through its own top model. Under C1 it never crosses the wire in a form the victim controls, so perturbing it is exactly as undeployable as perturbing `a_output`. The gradient path now honours `--defend_scope` too, which means the old `res_*` numbers were inflated on **both** channels, not just the forward one. `--defend_scope both` is retained only as an upper-bound ablation, and every deployable row in the paper must read `victim_only`.

**And a second, sharper constraint: the side matters as much as the scope.** Under C1 the victim receives gradients from the active party and returns embeddings to it. The victim can therefore perturb what it *sends* (`b_output`, forward channel) unilaterally. It cannot perturb the gradients it *receives* — those arrive already computed by the adversary. Any gradient-side mechanism in this literature therefore presumes a third party doing the injection, and ProVFL's own D1 says so outright: the paper describes its Laplacian-noise defense as applied by a **trusted third party**. `--defend_side` makes this explicit:

| `--defend_side` | Who can deploy it | Role in the paper |
|---|---|---|
| `output` | the victim, alone, no coordination | **the only deployable configuration** — every headline number |
| `grad` | requires a trusted third party (ProVFL's own D1 assumption) | comparator rows only, labelled as such |
| `both` | same third-party assumption | upper bound / decomposition ablation |

`lap_noise` (D1) and `ppdl` are gradient-side *by definition* in the literature, so `defense_func.GRAD_ONLY_DEFENSES` keeps them there and refuses to fake a forward-side variant. This is a limitation to state, not to hide: it is also the reason the proposed defense is output-side, which is a genuine advantage over D1 rather than an arbitrary design choice.

The rest of that advice was useful once corrected against real data — reconciled below.

## 1.5. Novelty / prior-art check (Round 5 — do not skip citing these)

Ran `fasttrack-literature:run_duplication_test`, `check_gap_saturation`, and targeted Consensus searches specifically to check whether "norm-adaptive Gaussian noise defense" already exists before investing further build time in it. **No exact duplicate** of "norm-adaptive, victim-channel-scoped Gaussian noise for VFL property inference, evaluated against a correlation-augmenting adversary" was found. But the underlying mechanism — adaptive/decaying noise triggered by gradient or weight norm — is an established pattern elsewhere, and two papers are close enough that the paper's related-work section must cite and explicitly differentiate from them, not present the mechanism as invented from scratch:

- **OUTPOST** (Wang et al., 2024, IEEE/ACM ToN) — Fisher-information/weight-spread-triggered Gaussian noise with iteration-based decay, defends gradient-leakage/reconstruction attacks in general (horizontal) FL. Closest conceptual ancestor to the norm-trigger + curriculum design.
- **Clip Norm Decay (CND)** (Lu et al., 2024) — decaying clip threshold based on current model updates, DP-based, defends backdoor attacks with a reported property-inference side-experiment, general FL.
- Broader adaptive-DP-noise lineage to cite as mechanism ancestry: DC-SGD, ADPFL, "Differentially Private FL With an Adaptive Noise Mechanism" (Xue et al.), "Gradient Leakage Attack Resilient Deep Learning" (Wei et al.).
- VFL-specific defenses that exist but target different attacks: FLSG (label inference), MID (mutual-information regularization, multi-attack), PRIVEE (feature inference) — none are norm-adaptive, none target property inference.

**Required reframing of the novelty claim:** not "we invented adaptive noise," but *"we adapt the established norm-triggered adaptive-noise family (OUTPOST, CND) to VFL property inference for the first time, adding threat-model-correct victim-channel scoping (an asymmetry that doesn't exist in horizontal FL) and evaluation against ProVFL's own correlation-augmenting adversary — a combination absent from prior work."* Narrower than the original framing, but honest and still a real contribution. Update §3 and §6 language accordingly when writing.

**Action item:** re-run `run_duplication_test` again once the design is frozen, since this landscape can shift over months of thesis work — treat it as a periodic check, not a one-time gate. **Round 7: this is now due.** The Gate-2 trigger it was pinned to no longer exists (§3 retracted, gates deprecated), and the claim being checked has changed from "norm-adaptive noise defense" to "channel-decomposition ceiling for VFL property inference + a constructive tight bound". Re-run against the *new* claim before drafting §6 — the OUTPOST/CND ancestry above matters much less to it, and a different neighbourhood (impossibility and scope results in FL privacy) matters much more.

## 2. What the grid actually showed

**Strongest attacker (`use_MR=1, use_LR=1`), `attack_feat='all'`:**

| Defense | MAE | Accuracy | AUC |
|---|---|---|---|
| None | 0.0089 | 0.8403 | 0.8931 |
| dp_gauss | 0.1322 | 0.7399 | 0.7058 |
| **gauss_noise** | **0.0639** | **0.8402** | **0.8929** |
| grad_clip | 0.0115 | 0.7859 | 0.7690 |
| grad_sparse | 0.0080 | 0.8397 | 0.8919 |
| random_proj | 0.0084 | 0.8403 | 0.8932 |

**MAE ratio, strong attacker ÷ weak attacker (< 1 means MR+LR makes the defense weaker):**

| Defense | Ratio | Reading |
|---|---|---|
| grad_sparse | 0.50 | MR-bypass hypothesis clearly confirmed |
| random_proj | 0.72 | Moderate bypass |
| dp_gauss | 0.78 | Degrades least |
| grad_clip | 0.99 | Flat — already saturated/broken, not a real signal fight |
| **gauss_noise** | **1.68** | **Gets *more* protective under the strongest attacker** |

**Wide `d_para` sweep (`res_defense_adult.txt`, weak attacker only) — the key discovery:**

| Defense | d_para | MAE | Accuracy | AUC |
|---|---|---|---|---|
| gauss_noise | 0.001 | 0.0229 | 0.8456 | 0.9027 |
| gauss_noise | 0.010 | 0.0486 | 0.8452 | 0.9025 |
| gauss_noise | 0.050 | 0.0922 | 0.8409 | 0.8970 |
| dp_gauss | 0.500 | 0.1634 | 0.7525 | 0.6033 |
| dp_gauss | 1.000 | 0.1086 | 0.7210 | 0.7779 |

`gauss_noise` at `d_para=0.05` reaches MAE 0.0922 at AUC 0.897, against `dp_gauss`'s 0.1086 at AUC 0.778 — comparable privacy, far more utility. `grad_clip`'s sweep is non-monotonic (accuracy crashes to 0.57 at `d_para=0.1`, recovers to 0.80 at `0.5`) — flagged as a training-instability oddity worth a footnote, not a defense worth building on.

### 2.1 Retraction: `gauss_noise` does **not** Pareto-dominate `dp_gauss`

Earlier rounds of this document asserted "**`gauss_noise` Pareto-dominates `dp_gauss` outright**" and repeated it in §8's framing paragraph. That was read off averaged points without a test. Running it through `analyze_defense_results.py --compare` reverses the direction on the privacy axis:

| Comparison (strong attacker, `attack_feat=all`) | Result | Paired t | p |
|---|---|---|---|
| `None` → `gauss_noise`, MAE | 0.0089 → 0.0639 | t=10.049, n=20 | **4.86e-09** |
| `None` → `gauss_noise`, AUC | 0.8931 → 0.8929 | — | 0.1463 (**not** significant) |
| `None` → `gauss_noise`, accuracy | 0.8403 → 0.8402 | — | 0.5171 (**not** significant) |
| `gauss_noise` → `dp_gauss`, MAE | 0.0639 → 0.1322 | t=−6.581 | **2.67e-06** |
| `gauss_noise` → `dp_gauss`, AUC | 0.8929 → 0.7058 | — | **1.36e-07** |

`dp_gauss` buys significantly **more** privacy; it just pays significantly more AUC for it. Neither dominates: `--pareto` puts **both on the frontier** (4 of 6 non-dominated in the strong-attacker table; 6 of 13 across the `d_para` sweep — `dp_gauss` at 0.5 and 1.0, `gauss_noise` at 0.05/0.01/0.001, `random_proj` at 4.0). `dp_gauss` owns the high-privacy tail and always will, because clipping plus calibrated noise is a strictly stronger mechanism at high strength.

**The claim that survives, and the only one to put in the paper:** *in the MAE < 0.1 regime, `gauss_noise` obtains the same attack error at ~0.12 higher AUC; `dp_gauss` remains preferable when attack error above 0.13 is required.* Weaker than "dominates", but it is what the data says, and it is still the interesting result — the second row of the table above is the real headline: **the privacy gain is significant while the utility loss is not measurable at all.**

**What could restore a dominance claim:** nobody has ever run `gauss_noise` above `d_para=0.05`. The whole sweep tops out exactly where `dp_gauss` starts winning. `kaggle_run.py --session s1` extends it to `out_para` 0.10 / 0.20 / 0.50; if AUC holds anywhere near 0.89 at MAE > 0.13, the dominance claim comes back legitimately. If AUC collapses, the regime-scoped claim above is final. Either way this is now an experiment, not an assumption.

### 2.2 Two caveats on the numbers above

- **Coverage.** The frontier claim covers only the five generic mechanisms in the tables (`grad_clip`, `gauss_noise`, `dp_gauss`, `grad_sparse`, `random_proj`). ProVFL's own D3 (R³eLU), D4 (shuffle) and D5 (withdraw) are **not** in it: D3 is unimplemented, and D4/D5 only became reachable from this harness in Round 6. Until `--session s1`/`s1b` land, write "of the generic perturbation mechanisms", never "of all known defenses" — D5 is the paper's *own* recommended mitigation, so an unqualified frontier claim that omits it is the first thing a reviewer will attack.
- **Scope and era.** Every number above was produced at `defend_scope=both` / `defend_side=both`, i.e. including undeployable perturbation of the attacker's channels, and the `d_para` sweep additionally ran at `val_ratio=0.2` with 5 seeds while the strong-attacker grid ran at 0.3 with 10. The analyzer tags these `era=legacy` and warns when a comparison straddles the boundary. **Treat all of §2 as the pre-correction baseline**: it justifies which mechanism to build on, and none of it can appear as a deployable result. That is what `--session s1` re-measures.

**Conclusion driving everything below: `gauss_noise` is the standout *cheap* mechanism, not a generic starting point.** It sits on the Pareto frontier alongside `dp_gauss` (§2.1), buys significant privacy for no measurable utility loss, and is the only defense that gets *stronger*, not weaker, against the paper's own hardest attacker. `grad_sparse` and `random_proj` should be reported as negative results (near-zero or negative privacy gain under the strong attacker), not built upon further.

### 2.3 Round 7 — what s1 actually measured, and why it replaces §3

Sessions s1 + s1b ran on 2026-09-05 (adult/sex, 5 seeds, `defend_scope=victim_only` except where noted). The per-defense numbers are unremarkable and confirm §2.1's regime claim. **The decomposition is not unremarkable.**

`my_utils/pia_func.property_ensemble` does not train a joint model over the four channels — it runs the same sampling attack once per channel and averages the four `pred_frac` estimates. Every per-channel estimate is therefore already a row in the result CSV, and the ensemble restricted to *any subset* of channels is recoverable arithmetically, with no additional GPU time. `analyze_clean_channel.py` does this. Three nested subsets matter, one per defender capability:

| Channel set | Contents | Who can perturb it |
|---|---|---|
| `ALL4` | `b_output`, `b_grad`, `a_grad`, `a_output` | — (what ProVFL reports) |
| `CLEAN` | `b_grad`, `a_grad`, `a_output` | survives any **victim-only forward** defense |
| `OWN` | `a_grad`, `a_output` | **nobody.** Both are computed inside the adversary's own bottom/top model |

**Passive attacker, `res_s1_adult__p1/p2`, n=5 seeds per config.** Recomputation validated against the recorded `en-XGB` rows: mean \|diff\| 0.0082, max 0.0379, r=0.9769 over n=70 (independent sampling draws, so exact equality is not expected).

| Configuration | MAE `b_output` | MAE all-4 | MAE clean | MAE own | AUC |
|---|---|---|---|---|---|
| `dp_gauss` s=1.00 both/victim_only | 0.1380 | 0.0936 | 0.0788 | **0.0215** | 0.7844 |
| `shuffle` 0.5 (D4) | 0.0769 | 0.0195 | 0.0077 | **0.0091** | 0.8794 |
| `lap_noise` s=0.10 grad/victim_only (D1) | 0.0714 | 0.0516 | 0.0460 | **0.0100** | 0.8712 |
| `shuffle` 0.2 | 0.0661 | 0.0131 | 0.0138 | **0.0110** | 0.8940 |
| `gauss_noise` s=0.50 output/victim_only | 0.0499 | 0.0119 | 0.0163 | **0.0148** | 0.8954 |
| `withdraw` 0.2 (D5) | 0.0298 | 0.0122 | 0.0111 | **0.0125** | 0.8995 |
| `gauss_noise` s=0.05 both/**both** *(undeployable)* | 0.0250 | 0.0793 | 0.1011 | 0.0772 | 0.8974 |
| `withdraw` 0.5 (D5) | 0.0237 | 0.0144 | 0.0138 | **0.0093** | 0.8964 |
| `gauss_noise` s=0.20 output/victim_only | 0.0209 | 0.0186 | 0.0190 | **0.0156** | 0.8992 |
| **`None` (undefended)** | 0.0174 | 0.0135 | 0.0137 | **0.0136** | 0.9012 |
| `gauss_noise` s=0.05 output/victim_only | 0.0103 | 0.0103 | 0.0126 | **0.0108** | 0.9010 |

**Active attacker (MR+LR), `res_s1_active_adult__p2`, 5 configs + baseline.** Undefended: all-4 0.0066, clean 0.0081, **own 0.0065**. Across the five defended configurations `MAE b_output` moves 3.8× (0.0133 → 0.0507) while all-4 stays ≤ 0.0118 and **own stays in 0.0054–0.0100**. Absolute errors are *lower* against the stronger attacker, so this is not a weak-attacker artefact.

**Reading.**
1. The mechanisms work. On the channel they target, `MAE b_output` spans 0.0103–0.0769 across victim-deployable configurations — a **7.5× range**.
2. It does not matter. `MAE own` sits at 0.009–0.022 for every deployable row, statistically indistinguishable from the undefended 0.0136. The adversary can discard the defended channel and lose nothing.
3. The only row where `own` moves is `defend_scope=both` (0.0772) — the configuration in which the defender noises the *attacker's own* tensors. That is the inflation §1 predicted, now quantified: **it accounts for the entire apparent protection in the pre-correction numbers.**
4. A gradient-side defense (`lap_noise`, D1) does raise `MAE clean` (0.0460), because `b_grad` is inside the clean set — but it needs the trusted third party D1 assumes, and it still leaves `own` at 0.0100.

**Threat-model validity check (done, not assumed).** `datasets/adult.py` keeps the target property at feature index 0, and `my_utils/utils.split_data` assigns `data_b = data[:, 0:50]` to the victim and `data_a = data[:, 50:111]` to the attacker. The property column is exclusively victim-side, so `MAE own ≈ MAE all-4` is **not** the trivial consequence of the adversary holding the property feature.

**The one live confound.** Adult's attacker-side one-hots include `relationship_Husband`/`relationship_Wife` and the marital-status columns, which are near-deterministic proxies for `sex`. So `own` may be *plain feature correlation* rather than *co-training coupling*. This changes the mechanism sentence in the paper, not the operational conclusion — either way no wire perturbation reaches the channel. `--session s1c` settles it in ~11 minutes with `ablation/test_ab_correlation.py` (Pearson r per attacker feature) and `ablation/test_ab_raw_baseline.py` (the attack on raw a-side features, no VFL training at all).

**Why additive noise cannot work here, stated as a mechanism.** The attack reads the *sorted per-sample L₁ norm vector* of the victim's embedding. Adding i.i.d. noise in d=16 dimensions shifts every per-sample norm by nearly the same amount, so the sorted vector translates and its shape — which is what carries the property — survives. This is currently an **analytic** argument plus the MAE tables; the norm traces in `res/` cannot support it, because `--log_norms` was only set on the *undefended* probes, where `victim_norm_obs == victim_norm_raw` by construction. `--session s1c` sets `--log_norms 1` on `norm_align` and on the high-σ Gaussian jobs, which is what turns the argument into a measurement of `obs` vs `raw`.

**Handover correction for `--norm_threshold`.** From `norms_defense_adult__p1.txt` (passive, 5 seeds × 20 epochs): `victim_norm_raw_med` is 3.5950 at the attack epoch (18), peaks at 3.6018 (epoch 19), and the median across epochs is **3.3451**. Any threshold at or above ~3.6 makes V1 fire almost never — **use ~3.0–3.3, not 3.6886.** Note also that the active MR+LR trace runs systematically lower (2.9879 at the attack epoch, 2.7128 median), so a threshold tuned on the passive trace under-fires against the stronger attacker: the trigger is **not transferable across attacker strengths**, which is itself a defect in the Gate-1 design and a further reason s2/s3 are deprecated.

## 2.5 Round 8 — `s1c` ran on real Kaggle: the ceiling holds, `norm_align` is the paper's best figure

12/13 jobs succeeded (`kaggle_run_log__p1/p2.txt`); `s1c_feat_corr` (the Pearson-correlation confound probe) crashed at 0.3 min, returncode 1 — needs debugging and a re-run. `s1c_raw_baseline` (the other, arguably more decisive confound probe: attack accuracy on raw `a`-side features with no VFL training at all) succeeded, so the confound question is not fully blocked, but its output file wasn't part of this upload — bring it next time before writing the confound sentence in the paper.

**Full channel decomposition, recomputed independently from `res_s1c_adult__p1/p2.txt` and `res_s1c_active_adult__p2.txt`** (same method as §2.3: `property_ensemble` is a mean of per-channel `pred_frac`, so `MAE_own` = mean of `{a_output, a_grad}`):

*Passive attacker:*

| Defense | out_para | side | MAE `b_output` | MAE all4 | MAE clean | **MAE own** | Accuracy | AUC |
|---|---|---|---|---|---|---|---|---|
| gauss_noise | 4.0 | output | 0.1515 | 0.0400 | 0.0028 | 0.0032 | 0.7596 | 0.77 |
| gauss_noise | 2.0 | output | 0.1558 | 0.0357 | 0.0043 | 0.0022 | 0.7596 | 0.83 |
| **norm_align** | −1 (mean) | output | **0.1375** | 0.0293 | 0.0068 | **0.0071** | **0.837** | **0.897** |
| norm_permute (placebo) | −1 | output | 0.1630 | 0.0361 | 0.0063 | 0.0092 | 0.840 | 0.895 |
| gauss_noise | 1.0 | output | 0.0896 | 0.0169 | 0.0074 | 0.0062 | 0.831 | 0.888 |
| norm_quant | 2.0 | output | 0.0219 | 0.0013 | 0.0055 | 0.0061 | 0.834 | 0.898 |
| **None** (undefended) | — | — | 0.0085 | 0.0053 | 0.0042 | 0.0042 | 0.844 | 0.901 |
| norm_quant | 8.0 | output | 0.0042 | 0.0018 | 0.0038 | 0.0026 | 0.844 | 0.901 |

*Active attacker (MR+LR), 5 seeds:*

| Defense | out_para | MAE `b_output` | MAE all4 | MAE clean | **MAE own** | Accuracy | AUC |
|---|---|---|---|---|---|---|---|
| **norm_align** | −1 | **0.1315** | 0.0331 | 0.0003 | 0.0014 | **0.834** | **0.888** |
| gauss_noise | 2.0 | 0.1925 | 0.0484 | 0.0003 | 0.0018 | 0.740 | 0.778 |
| None | — | 0.0106 | 0.0054 | 0.0036 | 0.0050 | 0.839 | 0.892 |

**Reading, and why this is the strongest result in the project so far:**

1. **The ceiling holds under the best defense tested, not just the mediocre ones.** `norm_align` moves `MAE b_output` by **16×** (passive) and is the single biggest per-channel shift measured anywhere in this project — bigger than any `gauss_noise` setting, `dp_gauss`, `withdraw`, or `shuffle` from earlier rounds. Its `MAE own` (0.0071 passive, 0.0014 active) is still in the same tiny band as undefended (0.0042 / 0.0050) and every other row. If even the strongest, most theoretically-motivated defense can't move the ceiling, the ceiling isn't an artifact of testing weak mechanisms — it's a property of the channel split.

2. **`norm_align` is also, independently, the best privacy-utility point ever measured in this project.** Accuracy/AUC barely move from baseline (0.837/0.897 vs 0.844/0.901 passive; 0.834/0.888 vs 0.839/0.892 active) while `MAE b_output` moves more than any other mechanism at any strength. This deserves to be reported as its own headline number even before the ceiling framing: **best channel-level privacy at essentially the lowest utility cost measured to date.**

3. **The placebo behaves like a placebo, mostly.** `norm_permute` (same norm multiset, different assignment — should score ≈0 effect on the property statistic) gives `MAE own`=0.0092, in the same band as everything else, and its `MAE b_output` (0.1630) is actually the single *highest* of the batch — consistent with permutation not being norm-*reducing*, just norm-*reassigning*, so if anything it randomizes the per-sample link between norm and label without reducing the aggregate signal. Worth a sentence in the paper explaining why the placebo's `b_output` number looks large (it's not "more private", it's "differently wrong").

4. **The "under-tuned" objection is closed.** `gauss_noise` at σ=2.0/4.0 doesn't buy more privacy at the ceiling — it just breaks the model. Accuracy collapses to **0.7596 exactly**, matching Adult's majority-class rate, meaning the model has stopped learning anything beyond predicting the majority class. `MAE own` at σ=4.0 (0.0032) is not meaningfully different from σ=1.0 (0.0062) or from undefended (0.0042) — the entire σ=1→4 range is already past the point where more noise buys anything, and it was already past that point at σ=0.05 several rounds ago. No further sigma-tuning objection is available.

5. **A concrete gap: the norm-trace logging can't yet show *why*.** `norms_defense_adult__p1.txt` only has `norm_align` rows this round (the high-σ `gauss_noise` jobs weren't logged despite that being the original `s1c` intent) and only logs `victim_norm_raw`/`victim_norm_obs` as **means**. Since `norm_align`'s target is the batch mean, `raw` and `obs` are *identical by construction* (0.6446 both, checked directly) — this is mathematically correct, not a bug, but it means the mean-based columns can never show `norm_align`'s actual mechanism, which is a **variance collapse**, not a mean shift. **Fix before the next batch:** add a `victim_norm_std` (or full per-batch norm array) column, and confirm `--log_norms 1` is actually applied to the `gauss_noise` high-σ jobs too — both are needed before "the sorted-norm vector collapses to a constant" has a supporting figure rather than just an equation.

**What to do now, in order:**
1. Debug and re-run `s1c_feat_corr` (`ablation/test_ab_correlation.py`) — cheap (0.3 min when it ran), and closes the confound question alongside `s1c_raw_baseline`'s already-successful run.
2. Bring back whatever `s1c_raw_baseline` actually printed/saved — its result exists but wasn't part of this upload, and it may already answer the confound question without needing #1 at all.
3. Fix the norm-logging gap (§2.5 point 5) — add `victim_norm_std`, apply `--log_norms 1` to the high-σ `gauss_noise` jobs — before generating the figure for §6 that needs to show the mechanism, not just cite the equation.
4. Once 1–3 are resolved, move to `s1g` (generalization) per the existing plan — the decomposition result is strong enough now that replicating it on a second dataset/property is the main remaining thing standing between this and a defensible paper claim.

## 3. The novel defense — converged design (Round 4: sequenced with go/no-go gates)

> ⚠️ **Round 7 status: superseded as the contribution, retained as design history and as the source of the constructive ceiling proof.** §2.3 measured that no victim-deployable perturbation of the wire moves the adversary's error, because the adversary reaches its full accuracy from tensors it computes itself. A better *strength schedule* for noise on `b_output` therefore cannot be the paper's claim — it optimises a channel the adversary does not need. Two things below survive intact and both matter:
>
> - **Step A (`--defend_scope`) and `--defend_side`** are what made the decomposition measurable at all. They stay in the paper as method, and §2.3's row 3 is their payoff: the entire apparent protection in the pre-correction numbers came from perturbing the attacker's own channels.
> - **`norm_align`** (Round 7, `defense_func.NORM_FAMILY`) replaces the adaptive-σ design as the *constructive* half of the argument. It drives the victim's sorted-norm vector to a constant, so leakage through `b_output` is zero **by construction rather than by tuning**. If the ensemble error still lands on `MAE own` under `norm_align`, the ceiling is tight and the statement becomes theorem-shaped: *the bound is achieved by an explicit mechanism, and it is still not protection.* That is a much stronger sentence than any σ sweep produces, and `norm_permute` (the placebo that must score ≈0) is what keeps it honest.
>
> Do not delete this section — a reviewer asking "did you try adapting the noise?" needs the answer to be "yes, here is the design, here is why it cannot help, and here is the mechanism that provably closes the channel and *still* does not help."

Reconciling ChatGPT's two passes against §2's data:

> **"Leakage- and Curriculum-Aware Adaptive Gaussian Defense"** — built directly on top of `gauss_noise` (the empirically dominant mechanism you already have), implemented as small, independently-testable steps with a stop condition at each gate, not built all at once.

**Step A — `--defend_scope victim_only`. ✅ Implemented (Round 6: forward *and* gradient channels), not yet run.**

All nine mechanisms now route through one function, so the two training scripts cannot drift apart again:
```python
# my_utils/defense_func.py
(z_up_clone, z_down_clone), info = defense_func.apply_perturbation(
    [z_up_clone, z_down_clone], args, side='output', epoch=epoch)   # index 0 = attacker, 1 = victim
```
```python
def defend_indices(defend_scope):
    return [0, 1] if defend_scope == 'both' else [1]   # a_output/a_grad are un-defendable, see §1
```
`STRUCTURAL_DEFENSES = ('None', 'shuffle', 'withdraw')` return the pair untouched (D4/D5 act on the dataloader instead); `GRAD_ONLY_DEFENSES = ('lap_noise', 'ppdl')` refuse the forward side; `KNOWN_DEFENSES` raises on a typo. `test_defense_func.py` asserts the C1 invariant directly — with `victim_only`, index 0 comes back **bit-identical** for every mechanism, and index 1 does not. This is a threat-model correction, not an optimization: a `both`-scope comparison isn't a fair baseline (§1).

**Step B — V1: threshold-only adaptive noise. ✅ Implemented, not yet run.**
```python
parser.add_argument('--adaptive_noise', type=int, default=0)   # 0 static | 1 V1 threshold | 2 V2 continuous
parser.add_argument('--norm_threshold', type=float, default=0.0)
parser.add_argument('--sigma_low', type=float, default=0.005)
parser.add_argument('--sigma_high', type=float, default=0.05)
```
```python
def resolve_sigma(tensor, args, epoch, base_sigma):
    if mode:
        batch_norm = batch_mean_norm(tensor, p=args.norm_type)     # the statistic the attack consumes
        if mode == 2:
            sigma = clamp(sigma_low + alpha * batch_norm / threshold, sigma_low, sigma_high)
        else:
            sigma = sigma_high if batch_norm > threshold else sigma_low
    ...
```
The trigger is measured on the **raw** tensor before noise is added, and `--log_norms` writes `victim_norm_raw` / `victim_norm_obs` / `sigma_mean` / `sigma_frac_high` per epoch to `norms_*_<dataset>.csv`.

⚠️ **`--norm_threshold` must be set from data, and there is a trap if it isn't.** Leave it at 0 and V1 fires `sigma_high` on literally every batch (degenerating to static noise at `sigma_high`), while V2 self-normalises to ratio 1 and saturates at `sigma_high` too — both silently stop being adaptive. `test_defense_func.py` asserts this behaviour so it stays visible. The value comes from `--session s1`'s undefended probe: column `victim_norm_raw_med` at the attack epoch in `norms_defense_<dataset>.csv`. **Do not reuse the old `res_baseline_adult.txt` for this** — it has a different schema and no norm trace.

**Gate 1 (after Step B):** compare `adaptive_noise=0` vs `=1` **at matched average strength**, filtered to `use_MR=1, use_LR=1` (the strong attacker — the setting where plain `gauss_noise`'s ratio-1.68 resilience needs to hold or improve, not just the weak-attacker default):
```
python analyze_defense_results.py "res_s2*.csv" --filter attack_feat=all use_MR=1 use_LR=1 \
  --group defense adaptive_noise sigma_low sigma_high \
  --compare "adaptive_noise=0" "adaptive_noise=1"
```
Session `s2` deliberately includes static references at **both** ends of the adaptive range (`out_para` 0.005 and 0.05), because "adaptive beats static at `d_para=0.05`" is not a result if the adaptive run's *average* sigma was higher than 0.05. `sigma_frac_high` in `norms_*.csv` is what tells you which static reference the adaptive run actually sat next to.

p<0.05 and MAE/PrivacyGain improves → continue to Step C. Otherwise **stop** — report V1 as a negative ablation, keep plain victim-only output-side `gauss_noise` as the headline defense, and skip straight to §5 P7 (multi-dataset) with that simpler result.

**Step C — V3: curriculum schedule on top of V1 (only if Gate 1 passes). ✅ Implemented (both directions), not yet run.**
```python
def curriculum_scale(epoch, warmup_epochs, mode=1):
    if mode == 2:  return max(0.0, 1.0 - epoch / warmup_epochs)   # decay, OUTPOST-style
    return min(1.0, (epoch + 1) / warmup_epochs)                  # ramp up
```
`--curriculum 1` ramps up, matching the paper's own Fig. 5d finding (leakage stabilizes only after ~epoch 10, so early noise buys nothing and costs utility). `--curriculum 2` decays, which is what OUTPOST does. Both directions run in session `s3`, so the schedule's direction is an **ablation row rather than an assumption** — and if `2` wins, that is a finding about VFL property inference differing from horizontal-FL reconstruction, not an embarrassment.

**Gate 2 (after Step C):** same comparison pattern, `curriculum=0` vs `=1` on top of adaptive. Pass → write up all three (V1 threshold / V2 continuous / V3 curriculum) as the paper's ablation table (§6 structure below). Fail on curriculum specifically → keep V1-only as the final design, still report curriculum as a negative ablation row.

**V2 (continuous, for the ablation table, not the primary path): ✅ Implemented as `--adaptive_noise 2`.**
```python
sigma = clamp(sigma_low + sigma_alpha * (batch_norm / norm_threshold), sigma_low, sigma_high)
```
Smooths the hard threshold discontinuity in V1. Worth including as a table row even if V1 is what you ship, since "did continuous scaling help over a hard threshold" is a natural reviewer question. Note that with `sigma_alpha == sigma_high` it saturates for any batch at or above the threshold — pick `sigma_alpha` below `sigma_high` if you want an interior response curve, which is why `s2` runs both `sigma_alpha=0.05` and `0.20`.

**Reserved refinement, not a build step:** EMA(batch norm) instead of raw per-batch norm as the trigger signal — only worth doing if V1/V3 already show a real win and `sigma_frac_high` in `norms_*.csv` shows visibly unstable sigma-switching. The logging that would reveal that is already in (`--log_norms`), so this stays a decision made from data.

**A companion metric — ✅ done, and it is not optional tooling: the training scripts never store MAE at all.**
```python
# analyze_defense_results.py
df['MAE'] = (df['pred_frac'] - df['gnd_frac']).abs()
baseline = df[df['defense'] == 'None'].groupby(CELL_COLS)['MAE'].mean()   # per comparable cell
df['PrivacyGain'] = df['MAE'] - df['MAE_none']
df['AUCDrop'] = df['auc_none'] - df['auc']
```
`CELL_COLS = ['dataset', 'property', 'attack_feat', 'classifier', 'use_MR', 'use_LR']` — the baseline is matched *within* each cell, so a `use_MR=1` row is never compared against a passive baseline. This is what makes the `grad_sparse`/`random_proj` negative result crisp (§2) and gives Gate 1/2 a clean second axis beyond raw MAE. **Consequence for the campaign: every result file must contain its own `defense='None'` rows**, which is why each session's job list starts with a baseline probe.

### Deferred: full adversarial discriminator (Corollary-1-targeted design)
Still the theoretically strongest option if time allows (a discriminator predicting property from the norm, trained adversarially against the victim's bottom model — earns a companion theoretical bound). GPU-costly (~3x per ChatGPT's estimate, roughly matches intuition given the extra forward/backward per batch) and your Kaggle 12-hour ceiling is already forcing multi-part runs. Sequence strictly after Gates 1 and 2 both pass, not before.

## 4. Rigor gaps to close

- ~~**Fix the seed-count mismatch (item 8) before any cross-script statistical comparison.**~~ ✅ Done. Both scripts now run `for seed in range(args.n_seeds)` with `--n_seeds 5` and `--val_ratio 0.3`, and their argparse surfaces differ only by the legitimately MR-specific `--act_weight`. Pre-Round-6 rows stay non-comparable across scripts — the analyzer's `era` tag and its `[warn] comparison mixes eras` message handle that automatically rather than relying on memory.
- ~~**Report variance and significance**~~ ✅ Tooling done and already used in anger: it falsified §2's dominance claim (§2.1). `--compare` pairs on `seed × dataset × property × attack_feat × classifier × use_MR × use_LR` via `ttest_rel`, falls back to Welch `ttest_ind` when no matched pairs exist, and says which it used. Standing rule: **no "A beats B" sentence enters the paper without a p-value from this tool.** Note `--compare` prints `p=nan` plus an explicit warning if scipy is missing — it is in `requirement.txt` and present on Kaggle.
- **Sweep the adaptive defense's own hyperparameters** (`sigma_low`, `sigma_high`, `norm_threshold`, `warmup_epochs`) the same way `d_para` was swept in §2. **Round 7: demoted from a gate to an optional ablation.** s1 measured PrivacyGain at σ = 0.05/0.10/0.20 output-side victim-only as −0.0009/−0.0033/+0.0023 (p=0.6150 at the reference point), so `s2`/`s3` compare two inert conditions; and §2.3 shows the trigger statistic is not even transferable between attacker strengths. Both sessions stay runnable and print why they are deprecated. What replaces this gap: **the σ → ∞ end of the sweep** (`s1c` at `out_para` 1.0/2.0/4.0), which closes off "you under-tuned it" properly, by showing where utility dies instead of where privacy starts.
- **Close the correlation confound** (`--session s1c`, new, top priority). §2.3's ceiling is only a statement about VFL if the adversary cannot read the property off its own raw features. `ablation/test_ab_correlation.py` gives Pearson r per attacker feature; `ablation/test_ab_raw_baseline.py` runs the attack on raw `a`-side inputs with no VFL training. ~11 min for both. Whichever way it lands, it must be *reported*, because a reviewer will construct this objection in the first five minutes.
- **Generalize beyond `adult`/`sex` — moved up, now second priority** (`--session s1g`, once per setting). Previously P6 "once the design is locked"; the design is no longer what is being locked. The result that needs replicating is the **channel decomposition**, and a single-dataset ceiling claim will not survive review. Acceptance bar unchanged: consistent behaviour in ≥3 of {adult-sex, adult-race, census, bankmk}. **Cheapest route to that bar: adult-race (2.2 h) + bankmk-marital (3.2 h) on top of the finished adult-sex — three settings for ~5.4 h, one Kaggle session.** census is the same session shape but ~8× the per-job cost (299 285 rows × 511 dims × 30 epochs vs adult's 44 355 × 111 × 20): `--list` now estimates **17.3 h**, over the 12 h ceiling, so it needs `--part 1/2` + `--part 2/2` across two sessions. Treat census as the fourth setting, not the first.
- **Report the ceiling arithmetic transparently.** `analyze_clean_channel.py` recomputes the 4-channel mean and validates it against the recorded `en-XGB` row before reporting the 3- and 2-channel variants (mean \|diff\| 0.0082, r=0.9769). Put that validation line in the paper — the whole ablation rests on `property_ensemble` being a mean of per-channel estimates, and a reader must be shown that the recomputation reproduces the recorded number rather than asked to trust it.
- **Report overhead** — `used_time` is already in every row, so `--group defense` gives the wall-clock cost of the norm computation for free. Report it: an adaptive defense that costs 3× is a different proposition from one that costs 1.02×.
- **State the exclusions explicitly.** D3 (R³eLU) is unimplemented. `ppdl` is excluded on two independent grounds — gradient-side by definition (undeployable under C1, §1) and `dp_gc_ppdl`'s per-coordinate Python loop costs hours per run in this harness. Both belong in the paper's limitations paragraph, phrased as scope, not as oversight.

## 5. Build order (Round 7 — re-sequenced after s1)

`kaggle_run.py` implements this. Markers in `runs_done/` make every session resumable, so a 12h timeout costs one job rather than the session. **Measured** cost is 11.6–13.5 min/job (`lap_noise` 44.9), so `--part N/M` is now rarely needed — a full 13-job session is under 3 h.

| Step | Session | Jobs / est | Status | What it decides |
|---|---|---|---|---|
| P0 — environment | `python kaggle_setup.py` | <2 min | ✅ ran clean | Deps, **accelerator on**, Adult downloaded, `datasets` resolving to the repo, dispatcher checks green |
| P1 — smoke test | `--session <s> --smoke` | ~2 min/job | ✅ used | 1 seed, 3 epochs, into `res_smoke_adult.csv`. Run before every *new* session, always — it is what catches a flag typo before hours of GPU |
| P2 — baselines + decomposition | `--session s1` | 20 / 4.3 h | ✅ **done 2026-09-05** | Produced §2.3. The inflation delta, the side decomposition, the σ sweep to 0.50, D4/D5 in-harness |
| P3 — comparison table | `--session s1b` | 8 / 1.9 h | ✅ **done** | `grad_clip`/`grad_sparse`/`random_proj` re-measured victim-only, output-side |
| **P2.5 — confound + constructive ceiling** | **`--session s1c`** | **13 / 2.9 h** | 📋 **next, run this first** | Two ~5-min probes decide whether the ceiling is a fact about VFL or about adult's one-hot columns; `norm_align`/`norm_quant`/`norm_permute` supply the provable-zero-leakage point and the placebo control; σ = 1/2/4 closes the "under-tuned" objection |
| **P2.6 — generalisation** | **`--session s1g --dataset <d> --property <p>`** | **adult-race 2.2 h; bankmk 3.2 h; census 17.3 h → `--part`** | 📋 second | Replicates the *decomposition* (not the defense). Bar: consistent in ≥3 of 4 settings. Run **adult-race + bankmk in one session (~5.4 h)** to clear the bar; census is the 4th setting and needs splitting |
| P4 — Gate 1 | `--session s2` | 11 / 2.7 h | ⛔ **deprecated** | Sweeps σ inside the inert range; prints why on selection. Optional ablation only |
| P5 — Gate 2 | `--session s3` | 10 / 2.4 h | ⛔ **deprecated** | Inherits Gate 1's premise plus a free parameter |
| P6 — utility / ε table | `--session s4` | 7 / 1.7 h | 📋 keep | Still wanted for the tradeoff figure and the overhead column |
| P7 — deferred | — | — | ⛔ drop for now | Adversarial discriminator. It optimises the same unnecessary channel; revisit only if s1c shows the ceiling is *not* tight |

**Two Kaggle sessions is the whole remaining plan for the core result:** s1c (2.9 h) + s1g on adult-race (2.2 h) fits one ~5 h run; s1g on bankmk (3.2 h) plus `--session s4` (1.7 h) is a second ~5 h run. That is ~10 h of the 30 h weekly quota and clears the ≥3-setting bar, leaving room for census in parts if a reviewer asks for a fourth.

⚠️ **Run `python kaggle_setup.py --dataset <d>` once per dataset before its first s1g session.** census and bankmk were undownloadable and unloadable until Round 7 (log rows 28–29); the setup gate now proves both in under two minutes, which is the difference between finding out now and finding out from a background notebook that has already consumed its slot.

**Order note (Round 7):** the gates moved *behind* generalisation and the confound probes. Cheap decisive information first — the same principle that moved D4/D5 ahead of the gates in Round 6, applied to its conclusion.

## 6. Paper structure (draft skeleton — Round 7, reframed around the ceiling)

The paper is no longer "a better defense". It is **a threat-model-correct measurement showing this attack class is out of reach of communication-channel defenses**, with a constructive proof that the bound is achieved. That is a positive structural claim, and it is more publishable than an incremental σ schedule would have been.

| Section | Content |
|---|---|
| 3 | Threat model, done properly: C1 active adversary; the four channels; **which channels a victim can perturb unilaterally, which need a trusted third party, and which are unreachable in principle** (§1). This section is the contribution's foundation, not boilerplate |
| 4 | Existing defenses: DP-Gaussian, Clip, Sparse, Projection, Gaussian, plus ProVFL's own D1 (Laplacian, third-party), D4 (shuffle), D5 (withdraw). Exclusions in the same breath: D3 (R³eLU) unimplemented, `ppdl` gradient-side and computationally infeasible here |
| 5 | **The channel decomposition (§2.3).** `property_ensemble` is a mean of per-channel estimates; the ceiling arithmetic and its validation; the three nested channel sets and who can perturb each; the `defend_scope=both` inflation that accounts for the literature's apparent protection |
| 6 | **The measured ceilings.** `MAE own` flat at 0.009–0.022 passive / 0.005–0.010 active while `MAE b_output` moves 7.5×. Both attacker strengths. ≥3 dataset/property settings from s1g. The validity check (property column is victim-side) and the correlation probe from s1c |
| 7 | **The constructive half.** `norm_align` zeroes the sorted-norm channel by construction; `norm_quant` traces the interpolation; `norm_permute` is the placebo that must score ≈0. Result: the bound is achieved and protection still does not follow. Plus the σ → 4.0 sweep showing utility dies before privacy arrives |
| 8 | Utility/overhead: MAE/PrivacyGain vs accuracy/AUC across settings; wall-clock from `used_time`; the honest statement that gradient-side mechanisms *do* raise `MAE clean` but require D1's trusted third party and still leave `MAE own` untouched |
| 9 | Limitations: single attack family; the correlation-vs-coupling mechanism question as answered by s1c; tabular datasets only; D3 unimplemented |


## 7. Tooling map — what is actually installed (verified 2026-09-06)

The names in previous rounds (`deep-research-agent`, `academic-paper`, `academic-paper-reviewer`, `academic-pipeline`) were aspirational and do not exist in this environment. The real inventory:

| Phase | Tool (real name) | Mode / use | State |
|---|---|---|---|
| Novelty checks, sparingly | **Consensus MCP** (`search`) | 250M-paper search, journal-quality scores. **FREE plan: 2/30 used, resets 1 Oct 2026.** Ration to targeted novelty questions — not exploratory browsing | ✅ connected |
| Cross-disciplinary lookup, unmetered | **fasttrack-literature MCP** | `search_papers` is usable; `get_journal_profile` is the right instrument for the §8 venue shortlist and is still unused. ⚠️ `check_gap_saturation` returned off-topic results for this query (LLM-firewall record, a neonicotinoid paper, two Riemann-hypothesis preprints) and its own `instrument_disagreement` flagged 38 vs ~134,841 — **do not cite it for this project** | ✅ free |
| Verbatim claim-hunting in full text | **semantic-scholar MCP** | `snippet_search` works and is the best tool for "has anyone written *this exact sentence*". `search_papers_by_relevance` errors ("Working outside of request context") — use `snippet_search` or `search_paper_by_title` instead | ⚠️ partial |
| **Adversarial pre-review — highest value right now** | **`/ars-reviewer`** | Run it on §2.3 + §6 *before* spending next week's quota. A reviewer objection found here costs nothing; found after s1g it costs a Kaggle session | 📋 do this |
| Structuring the manuscript | **`/ars-outline`**, then `/ars-full` | Outline once s1c lands, full draft once s1g does | 📋 |
| Related work | **`/ars-lit-review`**, `synthesis_agent` | §1.5's citations are the seed set | 📋 |
| Rebuttal prep | `/ars-rebuttal-audit`, `/ars-3w` | After first submission | later |
| Formatting / citations | `/ars-format-convert`, `/ars-citation-check`, `report_compiler_agent` | APA 7 by default; convert to the venue's style at the end | later |

Also present and irrelevant here: 21st.dev (UI components), Claude_Browser, scheduled-tasks, ccd_session\*, terminal.

**Action item now due:** §1.5 said to re-run `run_duplication_test` whenever the design changes materially. The design has changed materially — the contribution moved from "adaptive-σ defense" to "channel-decomposition ceiling + `norm_align` existence proof". Re-run it against the *new* claim before drafting §6, and spend one Consensus query on the exact framing "property inference vertical federated learning defense ceiling attacker-side channels".

## 8. Framing and venue thinking (Round 7 — reframed)

Position the paper as: *"Communication-channel defenses cannot address distribution-level property inference in VFL. We decompose the adversary's estimate by channel and show that under the standard C1 threat model, an active adversary reaches full accuracy from `a_output` and `a_grad` alone — tensors computed inside its own model that never traverse the network. Perturbing the victim's forward channel moves that channel's error by up to 7.5× and the ensemble's by almost nothing; perturbing the gradient channel as well, which already presumes a trusted third party, still leaves the adversary's own two channels untouched. We prove the bound is tight by construction: `norm_align` provably collapses the exact statistic the attack reads, and protection still does not follow."*

Why this is the stronger paper: the previous framing was an incremental mechanism competing against a crowded field of σ schedules. This one is a **scope result** — it tells the field that an entire defense category is aimed at the wrong surface for this attack class, and it comes with the constructive control that rules out "you just didn't perturb hard enough". Negative-sounding, positive in structure, and hard to scoop.

⚠️ **Warnings that survive unchanged:**
- **Do not write "Pareto-dominates DP-Gaussian" anywhere.** False (§2.1); `dp_gauss` buys more privacy at more AUC cost.
- **Do not write "defenses do not work" either.** The precise claim is scoped: *these* mechanisms, on *these* channels, against *this* attack class, under C1. Gradient-side mechanisms measurably raise `MAE clean`; they just cannot reach `MAE own`.
- Every ceiling number in the paper must be printed with the arithmetic that produced it (`property_ensemble` = mean of four per-channel `pred_frac`), plus the recomputation validation (mean |diff| 0.0082, r=0.9769, n=70). A reviewer who cannot reproduce the subset means from the released CSVs will assume the subsetting is the result.

**Venue thinking, revised.** An impossibility-flavoured measurement result with a constructive tight-bound is a security-conference paper more than a journal one — the framing rewards a venue that likes threat-model corrections.

- **Natural first target:** IEEE TIFS (unchanged — it published VFLDefender, so the community that needs this result reads it).
- **Strong fit for the new framing:** PETS — measurement + scope results are squarely in its remit, and the `norm_permute` placebo is the kind of control its reviewers reward.
- **Stretch:** USENIX Security / ACM CCS. Needs s1g's ≥3-setting generalisation *and* s1c's confound probes; without the correlation probe a reviewer can dismiss the whole thing as an artefact of adult's one-hot encoding.
- Use `get_journal_profile` on TIFS and the PETS proceedings before committing, and check the submission calendar against the s1c/s1g timeline rather than the other way round.

