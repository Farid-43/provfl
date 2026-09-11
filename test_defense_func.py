"""
Unit tests for the defense dispatcher. Pure tensor math on random inputs -- no
dataset, no model, no training, runs in under a second on CPU. The point is to
catch a mis-scoped or no-op defense here rather than after 12 hours of Kaggle.

  python test_defense_func.py
"""
import argparse
import sys

import torch

from my_utils import defense_func as D


def mk(defense='gauss_noise', **kw):
    a = argparse.Namespace(defense=defense, d_para=0.05, d_para2=1.0, out_para=-1.0,
                           defend_scope='victim_only', defend_side='both',
                           adaptive_noise=0, norm_threshold=0.0, sigma_low=0.005,
                           sigma_high=0.05, sigma_alpha=0.05, curriculum=0,
                           warmup_epochs=10, norm_type=1)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


FAILURES = []


def check(name, cond, detail=''):
    print('%-4s %s%s' % ('ok' if cond else 'FAIL', name, '  ' + detail if detail else ''))
    if not cond:
        FAILURES.append(name)


def pair():
    torch.manual_seed(0)
    return [torch.randn(64, 16), torch.randn(64, 16)]

def clone(p):
    return [t.clone() for t in p]


def same(x, y):
    return torch.equal(x, y)


ALL_DEFENSES = ['grad_clip', 'gauss_noise', 'dp_gauss', 'grad_sparse', 'random_proj',
                'lap_noise', 'ppdl']

# Strengths that are actually in range for each mechanism -- see test_param_traps for
# the values that silently turn one into a no-op.
PARA = {'grad_clip': 1.0, 'gauss_noise': 0.05, 'dp_gauss': 1.0, 'grad_sparse': 0.5,
        'random_proj': 4.0, 'lap_noise': 0.1, 'ppdl': 0.2}


def test_routing():
    check('defend_indices victim_only == [1]', D.defend_indices('victim_only') == [1])
    check('defend_indices both == [0,1]', D.defend_indices('both') == [0, 1])
    check('sides both', D.sides_enabled(mk(defend_side='both')) == ('output', 'grad'))
    check('sides output', D.sides_enabled(mk(defend_side='output')) == ('output',))
    check('sides grad', D.sides_enabled(mk(defend_side='grad')) == ('grad',))
    # out_para overrides d_para on the forward channel, but only when it is set (>=0)
    a = mk(d_para=0.05, out_para=0.20)
    check('_strength output uses out_para', D._strength(a, 'output') == 0.20)
    check('_strength grad uses d_para', D._strength(a, 'grad') == 0.05)
    a = mk(d_para=0.05, out_para=-1.0)
    check('_strength output falls back to d_para', D._strength(a, 'output') == 0.05)


def test_structural_noop():
    """None/shuffle/withdraw are realised in the dataloader or the input batch, so the
    dispatcher must return the pair untouched rather than quietly applying d_para."""
    for d in D.STRUCTURAL_DEFENSES:
        p = pair()
        ref = clone(p)
        for side in ('output', 'grad'):
            out, info = D.apply_perturbation(p, mk(d, defend_side='both'), side=side)
            check('%s is a no-op on %s' % (d, side),
                  same(out[0], ref[0]) and same(out[1], ref[1]) and info == {})

def test_scope_isolation():
    """The C1 invariant: with --defend_scope victim_only the attacker's own tensor must
    come back bit-identical, and the victim's must not. A defense that fails the second
    half is a no-op that would still have cost 22 minutes of Kaggle per run."""
    for d in ALL_DEFENSES:
        side = 'grad' if d in D.GRAD_ONLY_DEFENSES else 'output'
        p = pair()
        ref = clone(p)
        out, _ = D.apply_perturbation(p, mk(d, d_para=PARA[d], out_para=PARA[d],
                                            defend_scope='victim_only',
                                            defend_side=side), side=side)
        check('%s victim_only leaves the attacker untouched' % d, same(out[0], ref[0]))
        check('%s victim_only does perturb the victim' % d, not same(out[1], ref[1]))

    for d in ALL_DEFENSES:
        side = 'grad' if d in D.GRAD_ONLY_DEFENSES else 'output'
        p = pair()
        ref = clone(p)
        out, _ = D.apply_perturbation(p, mk(d, d_para=PARA[d], out_para=PARA[d],
                                            defend_scope='both',
                                            defend_side=side), side=side)
        check('%s scope both perturbs the attacker too' % d, not same(out[0], ref[0]))


def test_side_gating():
    for d in D.GRAD_ONLY_DEFENSES:
        p = pair()
        ref = clone(p)
        out, info = D.apply_perturbation(p, mk(d, d_para=PARA[d], defend_side='both'),
                                         side='output')
        check('%s never touches the forward channel' % d,
              same(out[1], ref[1]) and info == {})
    p = pair()
    ref = clone(p)
    out, _ = D.apply_perturbation(p, mk('gauss_noise', defend_side='output',
                                        out_para=0.05), side='grad')
    check('defend_side output ignores the grad call', same(out[1], ref[1]))
    p = pair()
    ref = clone(p)
    out, _ = D.apply_perturbation(p, mk('gauss_noise', defend_side='grad',
                                        d_para=0.05), side='output')
    check('defend_side grad ignores the output call', same(out[1], ref[1]))
    # the log_norms trace depends on info coming back from the victim channel only
    p = pair()
    out, info = D.apply_perturbation(p, mk('gauss_noise', defend_side='output',
                                           out_para=0.05, adaptive_noise=1,
                                           norm_threshold=1.0), side='output')
    check('gauss_noise reports sigma/norm for --log_norms',
          info.get('sigma') == 0.05 and info.get('norm') is not None)

def test_resolve_sigma():
    t = pair()[1]
    n = D.batch_mean_norm(t, p=1)
    check('batch_mean_norm is the per-sample L1 mean', 10.0 < n < 16.0, 'norm=%.3f' % n)
    std_n = D.batch_std_norm(t, p=1)
    check('batch_std_norm is positive for random inputs', std_n > 0.0, 'std=%.3f' % std_n)
    aligned_t = D.norm_alignment(t, target_norm=-1.0, p=1)
    std_aligned = D.batch_std_norm(aligned_t, p=1)
    check('batch_std_norm collapses to ~0 under norm_align', std_aligned < 1e-4, 'std=%.2e' % std_aligned)

    s, bn = D.resolve_sigma(t, mk(adaptive_noise=0), 0, 0.05)
    check('mode 0 is static and measures nothing', s == 0.05 and bn is None)

    # V1/V2 ignore base_sigma entirely, so 999 must not survive either branch
    s, bn = D.resolve_sigma(t, mk(adaptive_noise=1, norm_threshold=1.0), 0, 999.0)
    check('V1 above threshold -> sigma_high', s == 0.05 and bn is not None,
          'sigma=%.4f' % s)
    s, _ = D.resolve_sigma(t, mk(adaptive_noise=1, norm_threshold=1e6), 0, 999.0)
    check('V1 below threshold -> sigma_low', s == 0.005, 'sigma=%.4f' % s)

    s, _ = D.resolve_sigma(t, mk(adaptive_noise=2, norm_threshold=4 * n), 0, 999.0)
    check('V2 interpolates inside [low, high]',
          abs(s - (0.005 + 0.05 * 0.25)) < 1e-9, 'sigma=%.4f' % s)
    s, _ = D.resolve_sigma(t, mk(adaptive_noise=2, norm_threshold=0.0), 0, 999.0)
    check('V2 with --norm_threshold 0 pins sigma at sigma_high', s == 0.05,
          'it self-normalises to ratio 1, so the threshold must be set from norms_*.csv')


def test_curriculum():
    cs = D.curriculum_scale
    check('ramp starts at 1/w', cs(0, 10, 1) == 0.1)
    check('ramp reaches full strength at w-1', cs(9, 10, 1) == 1.0)
    check('ramp clamps at 1.0', cs(99, 10, 1) == 1.0)
    check('decay starts at 1.0', cs(0, 10, 2) == 1.0)
    check('decay halves at w/2', cs(5, 10, 2) == 0.5)
    check('decay floors at 0', cs(10, 10, 2) == 0.0 and cs(99, 10, 2) == 0.0)
    check('warmup_epochs 0 disables the schedule', cs(3, 0, 1) == 1.0)

    t = pair()[1]
    s, _ = D.resolve_sigma(t, mk(adaptive_noise=1, norm_threshold=1.0, curriculum=1,
                                 warmup_epochs=10), 0, 0.0)
    check('curriculum scales the V1 sigma', abs(s - 0.05 * 0.1) < 1e-12, 'sigma=%.5f' % s)
    s, _ = D.resolve_sigma(t, mk(adaptive_noise=0, curriculum=2, warmup_epochs=10),
                           5, 0.05)
    check('curriculum scales the static sigma', abs(s - 0.025) < 1e-12, 'sigma=%.5f' % s)

def test_mechanisms():
    t = pair()[1]
    c = D.gradient_clipping(t, max_norm=0.5)
    check('grad_clip bounds every per-sample L2 norm',
          float(c.norm(p=2, dim=1).max()) <= 0.5 + 1e-6)
    nz = (D.gradient_sparsification(t, keep_ratio=0.5) != 0).sum(dim=1)
    check('grad_sparse keeps exactly k coordinates per row',
          int(nz.min()) == 8 and int(nz.max()) == 8)
    check('gauss_noise at sigma 0 is an exact no-op',
          same(D.add_gaussian_noise(t, sigma=0.0), t))
    check('dp_gauss clips before it adds noise',
          not same(D.dp_gaussian_mechanism(t, 0.5, 0.0), t) and
          float(D.dp_gaussian_mechanism(t, 0.5, 0.0).norm(p=2, dim=1).max()) <= 0.5 + 1e-6)


def test_param_traps():
    """Strengths that silently disable a mechanism. Campaign footguns, not bugs --
    asserted here so they stay documented rather than being rediscovered from a run
    whose PrivacyGain came out at exactly 0.0000."""
    t = pair()[1]
    check('random_proj proj_dim >= hidden_dim is a no-op',
          same(D.gradient_random_projection(t, proj_dim=16), t))
    check('random_proj with d_para < 1 truncates to 0 and is a no-op',
          same(D.gradient_random_projection(t, proj_dim=0.05), t))
    check('random_proj proj_dim 4 does perturb',
          not same(D.gradient_random_projection(t, proj_dim=4.0), t))
    check('grad_sparse keep_ratio 1.0 is a no-op',
          same(D.gradient_sparsification(t, keep_ratio=1.0), t))
    check('grad_clip above the data norm is a no-op',
          same(D.gradient_clipping(t, max_norm=1e6), t))
    # a mistyped --defense used to silently produce undefended numbers under a
    # defense label; it must now fail on the first batch
    try:
        D.apply_perturbation(pair(), mk('guass_noise'), side='output')
        raised = False
    except ValueError:
        raised = True
    check('a mistyped --defense raises instead of no-opping', raised)
    check('every campaign defense name is known',
          all(d in D.KNOWN_DEFENSES for d in ALL_DEFENSES + list(D.STRUCTURAL_DEFENSES)))


def test_norm_family():
    """
    The norm-distribution defenses. Their whole justification is an exact claim
    about the statistic the attack reads -- the SORTED per-sample L_p norm vector
    -- so the claim is asserted numerically here rather than trusted.
    """
    t = pair()[1]

    # norm_align: the sorted-norm vector collapses to a constant, exactly.
    aligned = D.norm_alignment(t, target_norm=-1.0, p=1)
    n = aligned.norm(p=1, dim=1)
    check('norm_align makes every L1 norm equal',
          bool((n.max() - n.min()).abs().item() < 1e-4),
          'spread %.2e' % (n.max() - n.min()).abs().item())
    check('norm_align default target is the batch mean norm',
          abs(n.mean().item() - t.norm(p=1, dim=1).mean().item()) < 1e-3)
    check('norm_align honours an explicit target',
          abs(D.norm_alignment(t, target_norm=3.0, p=1).norm(p=1, dim=1).mean().item()
              - 3.0) < 1e-3)
    check('norm_align keeps directions (cosine 1 with the input)',
          bool(torch.nn.functional.cosine_similarity(aligned, t, dim=1).min().item()
               > 1 - 1e-5))
    # equalising the wrong order leaves a residual in the order the attack reads
    l2 = D.norm_alignment(t, target_norm=-1.0, p=2).norm(p=1, dim=1)
    check('norm_align at p=2 does NOT flatten the L1 spectrum',
          bool((l2.max() - l2.min()).item() > 0.5),
          'L1 spread %.3f -- p must match --norm_type' % (l2.max() - l2.min()).item())

    # norm_quant: k levels, and k=1 degenerates to alignment
    for k in (2, 4, 8):
        q = D.norm_quantization(t, num_levels=k, p=1).norm(p=1, dim=1)
        check('norm_quant k=%d yields <= %d distinct norms' % (k, k),
              len(torch.unique(torch.round(q * 1e2) / 1e2)) <= k,
              'got %d' % len(torch.unique(torch.round(q * 1e2) / 1e2)))
    check('norm_quant k=1 == norm_align',
          torch.allclose(D.norm_quantization(t, num_levels=1, p=1),
                         D.norm_alignment(t, target_norm=-1.0, p=1), atol=1e-6))

    # norm_permute: the placebo. Same multiset of norms => same sorted vector.
    permuted = D.norm_permutation(t, p=1)
    before = torch.sort(t.norm(p=1, dim=1)).values
    after = torch.sort(permuted.norm(p=1, dim=1)).values
    check('norm_permute leaves the SORTED norm vector unchanged (placebo)',
          bool(torch.allclose(before, after, atol=1e-4)),
          'max diff %.2e' % (before - after).abs().max().item())
    check('norm_permute does change the tensor (it is not a plain no-op)',
          not same(permuted, t))

    # gradients must still flow: these sit inside the forward graph
    leaf = torch.randn(64, 16, requires_grad=True)
    for name, fn in (('norm_align', lambda x: D.norm_alignment(x, -1.0, 1)),
                     ('norm_quant', lambda x: D.norm_quantization(x, 4, 1)),
                     ('norm_permute', lambda x: D.norm_permutation(x, 1))):
        leaf.grad = None
        fn(leaf).sum().backward()
        check('%s is differentiable' % name,
              leaf.grad is not None and bool(torch.isfinite(leaf.grad).all()))

    # routing: forward-side only, victim only, and refused on the gradient side
    for name in D.NORM_FAMILY:
        p0 = pair()
        out, _ = D.apply_perturbation(clone(p0), mk(name, out_para=-1.0), side='output')
        check('%s perturbs the victim on the output side' % name, not same(out[1], p0[1]))
        check('%s leaves the attacker untouched' % name, same(out[0], p0[0]))
        out, _ = D.apply_perturbation(clone(p0), mk(name), side='grad')
        check('%s is refused on the gradient side' % name,
              same(out[0], p0[0]) and same(out[1], p0[1]))
    check('the norm family is registered in KNOWN_DEFENSES',
          all(d in D.KNOWN_DEFENSES for d in D.NORM_FAMILY))

    # --out_para -1 must keep meaning "target the batch mean". _strength() would
    # fall back to --d_para (0.05) here, which would instead pin every embedding at
    # norm 0.05 -- a utility collapse unrelated to the norm spectrum, and the kind
    # of artefact that gets published as a privacy/utility tradeoff by mistake.
    p0 = pair()
    out, info = D.apply_perturbation(clone(p0), mk('norm_align', out_para=-1.0,
                                                   d_para=0.05), side='output')
    got = out[1].norm(p=1, dim=1).mean().item()
    want = p0[1].norm(p=1, dim=1).mean().item()
    check('norm_align --out_para -1 targets the batch mean, not d_para',
          abs(got - want) < 1e-3, 'got %.4f want %.4f' % (got, want))
    out, _ = D.apply_perturbation(clone(p0), mk('norm_align', out_para=3.0),
                                  side='output')
    check('norm_align honours an explicit --out_para through the dispatcher',
          abs(out[1].norm(p=1, dim=1).mean().item() - 3.0) < 1e-3)
    # a stray --d_para must not silently become the quantisation level either
    out, _ = D.apply_perturbation(clone(p0), mk('norm_quant', out_para=4.0,
                                               d_para=0.05), side='output')
    q = out[1].norm(p=1, dim=1)
    check('norm_quant reads the level from --out_para',
          len(torch.unique(torch.round(q * 1e2) / 1e2)) <= 4)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for fn in (test_routing, test_structural_noop, test_scope_isolation,
               test_side_gating, test_resolve_sigma, test_curriculum,
               test_mechanisms, test_param_traps, test_norm_family):
        print('\n-- %s' % fn.__name__)
        fn()
    print('\n%d checks failed%s'
          % (len(FAILURES), ': ' + ', '.join(FAILURES) if FAILURES else ''))
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
