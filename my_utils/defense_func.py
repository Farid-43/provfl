import torch
import numpy as np
import random
# reference: #https://github.com/FuChong-cyber/label-inference-attacks

# Privacy Preserving Deep Learning
def bound(grad, gamma):
    if grad < -gamma:
        return -gamma
    elif grad > gamma:
        return gamma
    else:
        return grad


def generate_lap_noise(beta):
    # beta = sensitivity / epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    # print(n_value)
    return n_value


def sigma(x, c, sensitivity):
    x = 2. * c * sensitivity / x
    return x


def get_grad_num(layer_grad_list):
    num_grad = 0
    num_grad_per_layer = []
    for grad_tensor in layer_grad_list:
        num_grad_this_layer = 0
        if len(grad_tensor.shape) == 1:
            num_grad_this_layer = grad_tensor.shape[0]
        elif len(grad_tensor.shape) == 2:
            num_grad_this_layer = grad_tensor.shape[0] * grad_tensor.shape[1]
        num_grad += num_grad_this_layer
        num_grad_per_layer.append(num_grad_this_layer)
    return num_grad, num_grad_per_layer


def get_grad_layer_id_by_grad_id(num_grad_per_layer, id):
    id_layer = 0
    id_temp = id
    for num_grad_this_layer in num_grad_per_layer:
        id_temp -= num_grad_this_layer
        if id_temp >= 0:
            id_layer += 1
        else:
            id_temp += num_grad_this_layer
            break
    return id_layer, id_temp


def get_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id):
    id_layer, id_in_this_layer = get_grad_layer_id_by_grad_id(num_grad_per_layer, id)
    grad_this_layer = layer_grad_list[id_layer]
    if len(grad_this_layer.shape) == 1:
        the_grad = grad_this_layer[id_in_this_layer]
    else:
        the_grad = grad_this_layer[id_in_this_layer // grad_this_layer.shape[1]][
            id_in_this_layer % grad_this_layer.shape[1]]
    return the_grad


def set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id, set_value):
    id_layer, id_in_this_layer = get_grad_layer_id_by_grad_id(num_grad_per_layer, id)
    grad_this_layer = layer_grad_list[id_layer]
    if len(grad_this_layer.shape) == 1:
        layer_grad_list[id_layer][id_in_this_layer] = set_value
    else:
        layer_grad_list[id_layer][id_in_this_layer // grad_this_layer.shape[1]][
            id_in_this_layer % grad_this_layer.shape[1]] = set_value


def dp_gc_ppdl(epsilon, sensitivity, layer_grad_list, theta_u, gamma, tau):
    grad_num, num_grad_per_layer = get_grad_num(layer_grad_list)
    c = int(theta_u * grad_num)
    # print("c:", c)
    # exit()
    epsilon1 = 8. / 9 * epsilon
    epsilon2 = 2. / 9 * epsilon
    used_grad_ids = []
    really_useful_grad_ids = []
    done_grad_count = 0
    while 1:
        r_tau = generate_lap_noise(sigma(epsilon1, c, sensitivity))
        while 1:
            while 1:
                grad_id = random.randint(0, grad_num - 1)
                if grad_id not in used_grad_ids:
                    used_grad_ids.append(grad_id)
                    break
                if len(used_grad_ids) == grad_num:
                    return
            grad = get_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, grad_id)
            r_w = generate_lap_noise(2 * sigma(epsilon1, c, sensitivity))
            if abs(bound(grad, gamma)) + r_w >= tau + r_tau:
                r_w_ = generate_lap_noise(sigma(epsilon2, c, sensitivity))
                set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, grad_id, bound((grad + r_w_), gamma))
                really_useful_grad_ids.append(grad_id)
                done_grad_count += 1
                if done_grad_count >= c:
                    for id in range(0, grad_num):
                        if id not in really_useful_grad_ids:
                            set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id, 0.)
                    # print("really_useful_grad_ids:", really_useful_grad_ids)
                    # print("len really_useful_grad_ids:", len(really_useful_grad_ids))
                    # exit()
                    return
                else:
                    break

# Differential Privacy(Noisy Gradients)
class DPLaplacianNoiseApplyer():
    def __init__(self, beta):
        self.beta = beta

    def noisy_count(self):
        # beta = sensitivity / epsilon
        beta = self.beta
        u1 = np.random.random()
        u2 = np.random.random()
        if u1 <= 0.5:
            n_value = -beta * np.log(1. - u2)
        else:
            n_value = beta * np.log(u2)
        n_value = torch.tensor(n_value)
        # print(n_value)
        return n_value

    def laplace_mech(self, tensor):
        # generate noisy mask
        # whether the tensor to process is on cuda devices
        noisy_mask = torch.zeros(tensor.shape).to(torch.float)
        if 'cuda' in str(tensor.device):
            noisy_mask = noisy_mask.cuda()
        noisy_mask = noisy_mask.flatten()
        for i in range(noisy_mask.shape[0]):
            noisy_mask[i] = self.noisy_count()
        noisy_mask = noisy_mask.reshape(tensor.shape)
        # print("noisy_tensor:", noisy_mask)
        tensor = tensor + noisy_mask
        return tensor


# ======================== New Defense Mechanisms ========================

def gradient_clipping(tensor, max_norm):
    """
    Per-sample gradient clipping (L2 norm).
    Clips each row (sample) of the gradient tensor so that its L2 norm <= max_norm.
    This bounds the sensitivity of individual samples, limiting information leakage
    through gradient magnitudes that the ProVFL attack exploits via L1-norm distributions.

    :param tensor: Gradient tensor of shape (batch_size, hidden_dim)
    :param max_norm: Maximum allowed L2 norm per sample
    :return: Clipped gradient tensor
    """
    norms = torch.norm(tensor, p=2, dim=1, keepdim=True)
    clip_factor = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
    return tensor * clip_factor


def add_gaussian_noise(tensor, sigma):
    """
    Add isotropic Gaussian noise N(0, sigma^2) to each element of the gradient tensor.
    This obscures the gradient distribution that ProVFL relies on for property inference.

    :param tensor: Gradient tensor of shape (batch_size, hidden_dim)
    :param sigma: Standard deviation of Gaussian noise
    :return: Noisy gradient tensor
    """
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise


def dp_gaussian_mechanism(tensor, max_norm, noise_multiplier):
    """
    Differentially Private Gaussian Mechanism (DP-SGD style, Abadi et al. 2016).
    Step 1: Per-sample gradient clipping to bound sensitivity to max_norm.
    Step 2: Add calibrated Gaussian noise with std = noise_multiplier * max_norm.
    Together these provide (epsilon, delta)-differential privacy guarantees.

    :param tensor: Gradient tensor of shape (batch_size, hidden_dim)
    :param max_norm: Clipping bound C for per-sample gradients
    :param noise_multiplier: Noise multiplier sigma; actual noise std = sigma * C
    :return: DP-protected gradient tensor
    """
    # Step 1: Per-sample gradient clipping
    clipped = gradient_clipping(tensor, max_norm)
    # Step 2: Add calibrated Gaussian noise
    noise_std = noise_multiplier * max_norm
    noise = torch.randn_like(clipped) * noise_std
    return clipped + noise


def gradient_sparsification(tensor, keep_ratio):
    """
    Gradient sparsification: retain only the top-k% of gradient coordinates by magnitude
    per sample, zeroing out the rest. This reduces the information content of communicated
    gradients and disrupts the sorted L1-norm distributions that ProVFL exploits.

    :param tensor: Gradient tensor of shape (batch_size, hidden_dim)
    :param keep_ratio: Fraction of coordinates to keep (0.0 to 1.0)
    :return: Sparsified gradient tensor
    """
    k = max(1, int(tensor.size(1) * keep_ratio))
    _, top_indices = torch.topk(torch.abs(tensor), k, dim=1)
    mask = torch.zeros_like(tensor)
    mask.scatter_(1, top_indices, 1.0)
    return tensor * mask


def gradient_random_projection(tensor, proj_dim):
    """
    Random Projection defense.
    Projects the gradient into a lower dimensional random subspace and then reconstructs it
    to the original dimension. This acts as privacy-preserving compression.

    :param tensor: Gradient tensor of shape (batch_size, hidden_dim)
    :param proj_dim: The dimension of the random subspace (int)
    :return: Obfuscated gradient tensor
    """
    hidden_dim = tensor.size(1)
    proj_dim = int(proj_dim)
    if proj_dim >= hidden_dim or proj_dim <= 0:
        return tensor

    # Generate a random Gaussian matrix
    R = torch.randn(hidden_dim, proj_dim, device=tensor.device) / (proj_dim ** 0.5)

    # Project to lower dimension and reconstruct
    projected = torch.matmul(tensor, R)
    reconstructed = torch.matmul(projected, R.t())

    return reconstructed


# ============ Norm-triggered adaptive Gaussian noise (proposed defense) ============

def batch_mean_norm(tensor, p=1):
    """
    Mean per-sample L_p norm of a batch. This is exactly the statistic ProVFL's
    attack consumes (sorted per-sample norms, see pia_func.construct_batch), so it
    is the natural trigger signal for an adaptive defense: measure the leakage
    channel itself rather than a proxy such as weight spread.

    :param tensor: (batch_size, hidden_dim) embedding or gradient tensor
    :param p: norm order, matching --norm_type (1 by default)
    :return: python float
    """
    return tensor.norm(p=p, dim=1).mean().item()


def curriculum_scale(epoch, warmup_epochs, mode=1):
    """
    Epoch-dependent multiplier on sigma.

    mode 1 (ramp-up): sigma reaches full strength over warmup_epochs. ProVFL's own
        Fig. 5d shows the attack's error only stabilises after ~epoch 10, so early
        epochs leak little that is usable; spending utility there is wasted.
    mode 2 (decay): the OUTPOST-style opposite -- strongest early, decaying after.
        Included so the schedule's direction is an ablation row rather than an
        untested assumption.

    :return: float in [0, 1]
    """
    w = int(warmup_epochs) if warmup_epochs else 0
    if w <= 0:
        return 1.0
    if int(mode) == 2:
        return max(0.0, 1.0 - float(epoch) / float(w))
    return min(1.0, float(epoch + 1) / float(w))


def resolve_sigma(tensor, args, epoch, base_sigma):
    """
    Gaussian sigma for one batch, on one channel.

    --adaptive_noise 0 : static, sigma = base_sigma (plain gauss_noise)
    --adaptive_noise 1 : V1, hard threshold -- sigma_high once the batch's mean
                         L_p norm exceeds --norm_threshold, sigma_low below it
    --adaptive_noise 2 : V2, continuous -- sigma_low + alpha * norm/threshold,
                         clamped to [sigma_low, sigma_high]; smooths V1's jump

    The result is then scaled by curriculum_scale() when --curriculum is set (V3).

    :return: (sigma, batch_norm) -- batch_norm is None when no norm was measured
    """
    mode = int(getattr(args, 'adaptive_noise', 0))
    sigma, batch_norm = base_sigma, None

    if mode:
        p = int(getattr(args, 'norm_type', 1))
        batch_norm = batch_mean_norm(tensor, p=p)
        threshold = float(getattr(args, 'norm_threshold', 0.0))
        sigma_low = float(getattr(args, 'sigma_low', 0.005))
        sigma_high = float(getattr(args, 'sigma_high', 0.05))
        if mode == 2:
            reference = threshold if threshold > 0 else max(batch_norm, 1e-8)
            alpha = float(getattr(args, 'sigma_alpha', 0.05))
            sigma = sigma_low + alpha * (batch_norm / reference)
            sigma = min(max(sigma, sigma_low), sigma_high)
        else:
            sigma = sigma_high if batch_norm > threshold else sigma_low

    curriculum = int(getattr(args, 'curriculum', 0))
    if curriculum:
        sigma = sigma * curriculum_scale(epoch, getattr(args, 'warmup_epochs', 10),
                                         mode=curriculum)
    return sigma, batch_norm


# ==================== Single dispatch point for all perturbations ====================
# Both vfl_pia_defense.py and vfl_pia_active.py route through apply_perturbation() so
# the two scripts cannot drift apart again (they already had divergent val_ratio and
# seed counts, which made their result files non-comparable).

# 'shuffle' perturbs the victim's raw input batch and 'withdraw' the aligned-sample
# set inside the dataloader, so neither is realised here.
STRUCTURAL_DEFENSES = ('None', 'shuffle', 'withdraw')

# D1 (Laplacian noise) and PPDL are defined in the literature as gradient-side
# mechanisms applied by a trusted third party; they stay on the gradient side.
GRAD_ONLY_DEFENSES = ('lap_noise', 'ppdl')

# --defense takes a free-form string in both scripts, so a typo used to fall through
# every branch below and produce undefended numbers filed under a defense name. Fail
# on the first batch instead of after 22 minutes of GPU time.
KNOWN_DEFENSES = STRUCTURAL_DEFENSES + GRAD_ONLY_DEFENSES + (
    'grad_clip', 'gauss_noise', 'dp_gauss', 'grad_sparse', 'random_proj')


def defend_indices(defend_scope):
    """
    Which channels a defender may touch. Index 0 = attacker (a), 1 = victim (b).

    victim_only: [1]. Under ProVFL's C1 threat model the adversary is the active
        party, so a_output and a_grad are the adversary's own local computation --
        no defender can force an adversarial party to perturb its own data.
    both: [0, 1]. Retained only as an (undeployable) upper-bound ablation.
    """
    return [0, 1] if defend_scope == 'both' else [1]


def sides_enabled(args):
    """--defend_side: 'output' (forward embeddings), 'grad' (returned gradients), 'both'."""
    side = getattr(args, 'defend_side', 'both')
    return ('output', 'grad') if side == 'both' else (side,)


def _strength(args, side):
    """Output side honours --out_para when set (>=0); the gradient side uses --d_para."""
    if side == 'output' and float(getattr(args, 'out_para', -1.0)) >= 0.0:
        return float(args.out_para)
    return float(args.d_para)


def apply_perturbation(pair, args, side, epoch=0):
    """
    Apply args.defense to a [attacker_tensor, victim_tensor] pair.

    :param pair: length-2 list/tuple, index 0 = attacker (a), index 1 = victim (b)
    :param side: 'output' or 'grad'
    :param epoch: current epoch, used by the curriculum schedule
    :return: (new_pair, info) where info carries the sigma and trigger norm actually
             used on the victim channel, for --log_norms. Empty dict when nothing ran.
    """
    defense = getattr(args, 'defense', 'None')
    out, info = list(pair), {}

    if defense not in KNOWN_DEFENSES:
        raise ValueError('unknown --defense %r; expected one of %s'
                         % (defense, ', '.join(KNOWN_DEFENSES)))
    if defense in STRUCTURAL_DEFENSES:
        return out, info
    if side not in sides_enabled(args):
        return out, info
    if side == 'output' and defense in GRAD_ONLY_DEFENSES:
        return out, info

    idx = defend_indices(getattr(args, 'defend_scope', 'victim_only'))
    para = _strength(args, side)

    if defense == 'gauss_noise':
        for i in idx:
            sigma, batch_norm = resolve_sigma(out[i], args, epoch, para)
            out[i] = add_gaussian_noise(out[i], sigma=sigma)
            if i == 1:
                info = {'sigma': sigma, 'norm': batch_norm}
    elif defense == 'grad_clip':
        for i in idx:
            out[i] = gradient_clipping(out[i], max_norm=para)
    elif defense == 'dp_gauss':
        for i in idx:
            out[i] = dp_gaussian_mechanism(out[i], max_norm=para,
                                           noise_multiplier=float(args.d_para2))
    elif defense == 'grad_sparse':
        for i in idx:
            out[i] = gradient_sparsification(out[i], keep_ratio=para)
    elif defense == 'random_proj':
        # gradient_random_projection reconstructs to the original dimension, so it is
        # safe on the forward side too. It used to be gradient-only, which silently
        # made it a no-op under --defend_side output.
        for i in idx:
            out[i] = gradient_random_projection(out[i], proj_dim=para)
    elif defense == 'lap_noise':
        dp = DPLaplacianNoiseApplyer(beta=para)
        for i in idx:
            out[i] = dp.laplace_mech(out[i])
    elif defense == 'ppdl':
        for i in idx:
            dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[out[i]],
                       theta_u=para, gamma=0.001, tau=0.0001)

    return out, info
