#libs

import time
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#experiment settings

MATRIX_SIZE = 256
N_ITER = 600
N_TRIALS = 100
SEED = 42

#support variables

SUBOPT_THRESHOLDS = [0.5, 0.1, 0.01]

MATMULS_PER_ITER = {
    "Newton-Schulz":      3,
    "RGD-QR":             3,
    "Landing":            4,
    "SGDM-Open":          9,
    "CayleySWATS-Naive":  14,
    "CayleySWATS-Angle":  14,
    "CayleySWATS-Proj":   14,
    "CayleySWATS-Hybrid": 14,
}

PLOT_ORDER = [
    "Newton-Schulz", "RGD-QR", "Landing", "SGDM-Open",
    "CayleySWATS-Naive", "CayleySWATS-Angle",
    "CayleySWATS-Proj",  "CayleySWATS-Hybrid",
]
COLORS = {
    "Newton-Schulz": "#1a1a2e",
    "RGD-QR": "#2E86AB",
    "Landing": "#57CC99",
    "SGDM-Open": "#F4A261",
    "CayleySWATS-Naive": "#4CC9F0",
    "CayleySWATS-Angle": "#06D6A0",
    "CayleySWATS-Proj":  "#E84855",
    "CayleySWATS-Hybrid": "#9B5DE5",
}

#main hyperparams, more sustantially since paper fine tuned to n=256

LR = 0.05
BETA = 0.85
S = 3

BETA1 = 0.9
BETA2 = 0.995
EPS_ADAM = 1e-8

EPS_NAIVE = 3e-3
EPS_PROJ = 3e-5
EPS_ANGLE = 0.06
EPS_CURV = 0.06

CONSEC_NAIVE = 4
CONSEC_ANGLE = 8
CONSEC_PROJ = 4
CONSEC_HYBRID = 7

LR_LANDING = 0.04
GAMMA = 0.90
LAM = 1.0

#support funcs for speeding up calculations, optimized by Claude Code, not author(!!!!)

def skew_part(M):
    if not np.all(np.isfinite(M)):
        return np.zeros_like(M)
    return (M - M.T) * 0.5

def sym_part(M):
    return (M + M.T) * 0.5

def frob(M):
    return float(np.sqrt(np.einsum("ij,ij->", M, M)))

def mat_inner(A, B):
    return float(np.einsum("ij,ij->", A, B))

def euclidean_grad(Q, A):
    return 2.0 * (Q - A)

def project_to_tangent_space(G, Q):
    return G - Q @ sym_part(Q.T @ G)

def riemannian_grad(Q, A):
    return project_to_tangent_space(euclidean_grad(Q, A), Q)

def orthogonality_defect(X):
    n = X.shape[0]
    D = X @ X.T
    D.flat[::n + 1] -= 1.0
    return 0.25 * float(np.einsum('ij,ij->', D, D))

def orthogonality_penalty_grad(X, I):
    if frob(X) < 1e-6:
        return np.zeros_like(X)
    g  = (X @ X.T - I) @ X
    gn = frob(g)
    if not np.all(np.isfinite(g)) or gn < 1e-30:
        return np.zeros_like(X)
    return g / (gn + 1e-8)

def qr_retraction(Y):
    Q, R  = np.linalg.qr(Y)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs

def cayley_retraction(Q, momentum, lr, s=3):
    W_hat = momentum @ Q.T - 0.5 * Q @ (Q.T @ momentum @ Q.T)
    W     = W_hat - W_hat.T

    alpha = min(lr, 1.0 / (frob(W) + 1e-8))
    WQ    = W @ Q

    h = alpha * 0.5
    Y = Q + alpha * WQ
    for _ in range(s):
        Y = Q + h * (W @ (Q + Y))

    return Y, WQ

def loss(X, A):
    D = A - X
    return float(np.einsum("ij,ij->", D, D))

def rel_subopt(f, f_star, f0):
    return max((f - f_star) / (f0 - f_star + 1e-30), 1e-15)

#direct solvers

def optimal_solution(A):
    U, _, Vt = np.linalg.svd(A)
    Q_star   = U @ Vt
    return Q_star, loss(Q_star, A)

def newton_schulz(A, n_iter=30):
    n = A.shape[0]

    v = np.ones(n) / np.sqrt(n)
    for _ in range(5):
        v = A   @ v;  v /= (np.linalg.norm(v) + 1e-30)
        v = A.T @ v;  v /= (np.linalg.norm(v) + 1e-30)
    sigma_max = float(np.linalg.norm(A @ v)) * 1.02

    X = A / (sigma_max + 1e-8)

    loss_hist   = [loss(X, A)]
    defect_hist = [orthogonality_defect(X)]

    t0 = time.perf_counter()
    for _ in range(n_iter):
        X = 1.5 * X - 0.5 * (X @ X.T @ X)
        loss_hist.append(loss(X, A))
        defect_hist.append(orthogonality_defect(X))
        if defect_hist[-1] < 1e-13:
            break
    elapsed = time.perf_counter() - t0

    while len(loss_hist) < n_iter + 1:
        loss_hist.append(loss_hist[-1])
        defect_hist.append(defect_hist[-1])

    return X, np.array(loss_hist[:n_iter+1]), np.array(defect_hist[:n_iter+1]), elapsed

#current optimizer state classes

class RGDState:
    def __init__(self, Q):
        self.Q = Q

class LandingState:
    def __init__(self, X, mom):
        self.X = X
        self.mom = mom

class SGDMState:
    def __init__(self, Q, mom):
        self.Q = Q
        self.mom = mom

class CayleySWATSState:
    def __init__(self, Q, n):
        self.Q = Q
        self.adam_m = np.zeros((n, n))
        self.adam_v = np.zeros((n, n))
        self.adam_t = 0
        self.sgdm_mom = np.zeros((n, n))
        self.phase = 1
        self.crit_prev = 0.0
        self.consec = 0
        self.grad_prev = None

#optimizers step funcs

def get_current_point(state):
    if isinstance(state, LandingState):
        return state.X
    return state.Q

def step_rgd(state, A):
    grad = riemannian_grad(state.Q, A)
    Q_new = qr_retraction(state.Q - LR * grad)
    return RGDState(Q=Q_new)

def step_landing(state, A, I):
    skew_grad = skew_part(euclidean_grad(state.X, A) @ state.X.T)
    if not np.all(np.isfinite(skew_grad)):
        skew_grad = np.zeros_like(state.X)

    new_mom = (1.0 - GAMMA) * state.mom + GAMMA * skew_grad
    pen = orthogonality_penalty_grad(state.X, I)
    X_new = state.X - LR_LANDING * (new_mom @ state.X + LAM * pen)

    if not np.isfinite(frob(X_new)) or frob(X_new) < 1e-6:
        X_new = state.X.copy()

    return LandingState(X=X_new, mom=new_mom)

def step_sgdm(state, A):
    grad = euclidean_grad(state.Q, A)
    new_mom = BETA * state.mom - grad
    Q_new, _ = cayley_retraction(state.Q, new_mom, LR, S)
    return SGDMState(Q=Q_new, mom=new_mom)

def adam_step(state, A):
    t_new = state.adam_t + 1
    grad = riemannian_grad(state.Q, A)

    m_new = BETA1 * state.adam_m + (1.0 - BETA1) * grad
    v_new = BETA2 * state.adam_v + (1.0 - BETA2) * (grad * grad)

    m_hat = m_new / (1.0 - BETA1 ** t_new)
    v_hat = v_new / (1.0 - BETA2 ** t_new)

    U = m_hat / (np.sqrt(v_hat) + EPS_ADAM)

    Q_new, _ = cayley_retraction(state.Q, -U, LR, S)
    return Q_new, grad, U, m_new, v_new, t_new

#swats swtich criterion func

def check_switch_criterion(state, grad, U, new_m, variant):
    grad_prev = state.grad_prev if state.grad_prev is not None else grad

    if variant == "naive":
        lam = mat_inner(U, grad) / (frob(grad) ** 2 + 1e-30)
        triggered = abs(lam - state.crit_prev) < EPS_NAIVE
        store = lam

    elif variant == "angle":
        nu, nm = frob(U), frob(new_m)
        if nu < 1e-30 or nm < 1e-30:
            triggered = False
        else:
            cos = np.clip(mat_inner(U, new_m) / (nu * nm), -1.0, 1.0)
            triggered = float(np.arccos(cos)) < EPS_ANGLE
        store = state.crit_prev

    elif variant == "proj":
        ratio = frob(U) / (frob(grad) + 1e-8)
        triggered = abs(ratio - state.crit_prev) < EPS_PROJ
        store = ratio

    else:
        lam = mat_inner(U, grad) / (frob(grad) ** 2 + 1e-30)
        rel_grad_diff = frob(grad - grad_prev) / (frob(grad) + 1e-8)
        triggered = (abs(lam - state.crit_prev) < EPS_NAIVE) and \
                        (rel_grad_diff < EPS_CURV)
        store = lam

    new_consec = state.consec + 1 if triggered else 0
    return new_consec, store

def step_cayley_swats(state, A, variant, consec_needed):
    new_state = CayleySWATSState.__new__(CayleySWATSState)
    new_state.phase = state.phase

    if state.phase == 1:
        Q_new, grad, U, m_new, v_new, t_new = adam_step(state, A)

        new_consec, store = check_switch_criterion(state, grad, U, m_new, variant)
        switch_now = (new_consec >= consec_needed)

        new_state.Q = Q_new
        new_state.adam_m = m_new
        new_state.adam_v = v_new
        new_state.adam_t = t_new
        new_state.sgdm_mom = m_new.copy() if switch_now else state.sgdm_mom
        new_state.phase = 2 if switch_now else 1
        new_state.crit_prev = store
        new_state.consec = new_consec
        new_state.grad_prev = grad

    else:
        grad = euclidean_grad(state.Q, A)
        new_mom = BETA * state.sgdm_mom - grad
        Q_new, _ = cayley_retraction(state.Q, new_mom, LR, S)

        new_state.Q = Q_new
        new_state.adam_m = state.adam_m
        new_state.adam_v = state.adam_v
        new_state.adam_t = state.adam_t
        new_state.sgdm_mom = new_mom
        new_state.crit_prev = state.crit_prev
        new_state.consec = state.consec
        new_state.grad_prev = state.grad_prev

    return new_state

#get matmuls

def estimate_matmuls(name, n_iter, phase_switch):
    if phase_switch is not None and name.startswith("CayleySWATS"):
        return phase_switch * 14 + (n_iter - phase_switch) * 9
    return n_iter * MATMULS_PER_ITER.get(name, 9)

def run_optimizer(name, step_fn, init_state, A, f_star, n_iter,
                  project_at_end=False):
    state = init_state
    pt = get_current_point(state)
    f0 = loss(pt, A)

    losses = [f0]
    defects = [orthogonality_defect(pt)]
    phase_switch = None
    iters_to = {eps: None for eps in SUBOPT_THRESHOLDS}

    t_start = time.perf_counter()
    for i in range(n_iter):
        prev_phase = getattr(state, 'phase', None)
        state = step_fn(state, A)
        cur_phase = getattr(state, "phase", None)

        if prev_phase == 1 and cur_phase == 2 and phase_switch is None:
            phase_switch = i + 1

        pt = get_current_point(state)
        losses.append(loss(pt, A))
        defects.append(orthogonality_defect(pt))

        r = rel_subopt(losses[-1], f_star, f0)
        for eps in SUBOPT_THRESHOLDS:
            if iters_to[eps] is None and r < eps:
                iters_to[eps] = i + 1

    elapsed = time.perf_counter() - t_start

    if project_at_end:
        pt_final    = qr_retraction(get_current_point(state))
        losses[-1]  = loss(pt_final, A)
        defects[-1] = orthogonality_defect(pt_final)

    losses_arr = np.array(losses)
    return {
        "losses":       losses_arr,
        "rel_subopt":   np.array([rel_subopt(l, f_star, f0) for l in losses_arr]),
        "defects":      np.array(defects),
        "time":         elapsed,
        "phase_switch": phase_switch,
        "f0":           f0,
        "iters_to":     iters_to,
        "matmuls":      estimate_matmuls(name, n_iter, phase_switch)}

def make_all_optimizers(n, Q0, n_iter):

    def swats(variant, consec):
        st = CayleySWATSState(Q=Q0.copy(), n=n)
        fn = partial(step_cayley_swats, variant=variant, consec_needed=consec)
        return "CayleySWATS-" + variant.capitalize(), fn, st, False

    return [
        ("RGD-QR", partial(step_rgd), RGDState(Q=Q0.copy()), False),
        ("Landing", partial(step_landing, I=np.eye(n)), LandingState(X=Q0.copy(), mom=np.zeros((n,n))), True),
        ("SGDM-Open", partial(step_sgdm), SGDMState(Q=Q0.copy(), mom=np.zeros((n,n))), False),
        swats("naive",  CONSEC_NAIVE),
        swats("angle",  CONSEC_ANGLE),
        swats("proj",   CONSEC_PROJ),
        swats("hybrid", CONSEC_HYBRID)]

#plots by Claude Code, not author(!!!!!!)

def draw_bar_chart(ax, names, values, title, ylabel,
                   log_scale=False, fmt="{:.3f}", budget_line=None):
    xp   = np.arange(len(names))
    vals = [max(abs(values.get(nm, float('nan'))), 1e-10)
            if np.isfinite(values.get(nm, float('nan'))) else float('nan')
            for nm in names]
    disp = [v if np.isfinite(v) else 0 for v in vals]

    bars = ax.bar(xp, disp,
                  color=[COLORS.get(nm, "#aaa") for nm in names],
                  width=0.6, edgecolor='white', linewidth=1.2, zorder=3)

    if log_scale:
        ax.set_yscale('log')
    if budget_line is not None:
        ax.axhline(budget_line, color="#888", ls="--", lw=1.0,
                   label=f"budget = {budget_line}")
        ax.legend(fontsize=8)

    finite = [v for v in vals if np.isfinite(v) and v > 0]
    if finite:
        ax.set_ylim(1e-10 if log_scale else 0,
                    max(finite) * (4 if log_scale else 1.38))

    ax.set_xticks(xp)
    ax.set_xticklabels(names, fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(axis='y', alpha=0.2, lw=0.6, which='both' if log_scale else 'major')

    for bar, v in zip(bars, vals):
        if not np.isfinite(v) or v <= 0:
            continue
        ypos = v * 1.5 if log_scale else v + (max(finite) * 0.025 if finite else 0)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                fmt.format(v), ha='center', va='bottom', fontsize=7.0)

def make_plots(avg_subopt, avg_iters_to, avg_matmuls, n, n_iter, n_trials):
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    fig.patch.set_facecolor("#F8F9FA")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")

    fig.suptitle(
        f"Orthogonal Approximation  $\\min_{{Q\\in O(n)}}\\|A-Q\\|_F^2$"
        f"   ($n={n}$, budget={n_iter}, {n_trials} trials)",
        fontsize=12, y=1.01)

    names    = [nm for nm in PLOT_ORDER if nm in avg_subopt]
    eps_show = 0.1

    iters_at = {nm: (avg_iters_to.get(nm, {}).get(eps_show) or n_iter) for nm in names}

    draw_bar_chart(axes[0], names, iters_at,
                   f"Iters to rel_subopt < {eps_show}  ({n_trials} trials avg)",
                   "Iters", log_scale=False, fmt="{:.0f}", budget_line=n_iter)

    draw_bar_chart(axes[1], names, avg_matmuls,
                   f"Matmul count  ({n_trials} trials avg)",
                   "Matmuls", log_scale=False, fmt="{:.0f}")

    draw_bar_chart(axes[2], names, avg_subopt,
                   f"Rel suboptimality  ({n_trials} trials avg)",
                   r"$(f-f^*)\,/\,(f_0-f^*)$  [log]",
                   log_scale=True, fmt="{:.2e}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

#generate and standartize random matrix

def make_problem(rng, n):
    A  = rng.standard_normal((n, n))
    A /= (np.linalg.norm(A, "fro") / np.sqrt(n))
    Q0, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return A, Q0

#run set trials

def main():
    n = MATRIX_SIZE
    rng = np.random.default_rng(SEED)

    all_subopt   = {nm: [] for nm in PLOT_ORDER}
    all_iters_to = {nm: [] for nm in PLOT_ORDER}
    all_matmuls  = {nm: [] for nm in PLOT_ORDER}

    for trial in range(N_TRIALS):
        rng_t = np.random.default_rng(trial + 7)
        A_t, Q0_t = make_problem(rng_t, n)
        _, fs_t = optimal_solution(A_t)

        _, ns_losses, _, _ = newton_schulz(A_t, n_iter=N_ITER)
        ns_f0  = ns_losses[0]
        ns_rel = [rel_subopt(l, fs_t, ns_f0) for l in ns_losses]
        all_subopt["Newton-Schulz"].append(rel_subopt(ns_losses[-1], fs_t, ns_f0))
        all_iters_to["Newton-Schulz"].append({
            eps: next((i for i, r in enumerate(ns_rel) if r < eps), None)
            for eps in SUBOPT_THRESHOLDS
        })
        all_matmuls["Newton-Schulz"].append(
            len(ns_losses) * MATMULS_PER_ITER["Newton-Schulz"])

        for name, step_fn, init_st, proj_end in make_all_optimizers(n, Q0_t, N_ITER):
            res  = run_optimizer(name, step_fn, init_st, A_t, fs_t, N_ITER,
                                 project_at_end=proj_end)
            f0_t = res["f0"]
            all_subopt[name].append(rel_subopt(res["losses"][-1], fs_t, f0_t))
            all_iters_to[name].append(res["iters_to"])
            all_matmuls[name].append(res["matmuls"])

        print(f"trial {trial + 1}/{N_TRIALS} done")

    avg_subopt  = {nm: float(np.mean(all_subopt[nm]))
                   for nm in PLOT_ORDER if all_subopt[nm]}
    avg_matmuls = {nm: float(np.mean(all_matmuls[nm]))
                   for nm in PLOT_ORDER if all_matmuls[nm]}
    avg_iters_to = {
        nm: {eps: float(np.mean([(d[eps] if d[eps] is not None else N_ITER)
                                  for d in all_iters_to[nm]]))
             for eps in SUBOPT_THRESHOLDS}
        for nm in PLOT_ORDER if all_iters_to[nm]}

    print("____________________________________________________________")
    for nm in PLOT_ORDER:
        if nm not in avg_subopt:
            continue
        it01 = avg_iters_to.get(nm, {}).get(0.1, N_ITER)
        print(f"  {nm:26s}  subopt={avg_subopt[nm]:.3f}  iters@0.1={it01:.03f}  matmuls={avg_matmuls.get(nm, 0):.03f}")

    make_plots(avg_subopt, avg_iters_to, avg_matmuls, n, N_ITER, N_TRIALS)


if __name__ == "__main__":
    main()
