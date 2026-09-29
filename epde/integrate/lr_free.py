"""Optimisers with no learning rate, for the DeepXDE fitness solve.

Each one takes the length scale a step size carries from somewhere other than a
tuned constant:

* ``DoG`` -- distance over gradients (Ivgi, Hinder & Carmon, ICML 2023):
  eta_t = rbar_t / sqrt(sum_{i<=t} ||g_i||^2), rbar_t = max(r_eps, max ||x_i - x_0||).
  The one constant is ``reps_rel`` (r_eps = reps_rel * (1 + ||x_0||)), a
  relative start that the method grows out of. Global (not per-layer) and the
  LAST iterate: the paper's iterate averaging is a separate mechanism.
* ``Prodigy`` -- Adam whose step is D-adapted (Mishchenko & Defazio, ICML 2024,
  Algorithm 1 as in the reference ``prodigyopt`` code), re-implemented here so
  no package is installed. Pinned to Adam's betas/eps with bias correction on,
  ``d_coef = 1`` (it would be an lr in disguise), no weight decay, unbounded
  growth. The one constant is ``d0``.
* ``levenberg_marquardt`` -- damped Gauss-Newton on the stacked residual
  vector, the loss being a sum of squares. The damping adapts by Nielsen's
  gain-ratio rule from ``lambda_0 = tau * max diag(J^T J)``; ``tau`` is the one
  constant, and it is relative to the curvature.

None of this is imported unless an adapter asks for it.
"""
import math
import time

import torch


class DoG(torch.optim.Optimizer):
    """Distance-over-Gradients SGD, global version, last iterate."""

    def __init__(self, params, reps_rel: float = 1e-6):
        if not reps_rel > 0.0:
            raise ValueError(f"reps_rel must be > 0, got {reps_rel!r}")
        super().__init__(params, {"reps_rel": float(reps_rel)})
        self._x0 = None
        self._rbar = None
        self._g2 = 0.0
        self.last_eta = None

    def _flat(self, attr):
        out = []
        for group in self.param_groups:
            for p in group["params"]:
                t = p if attr is None else p.grad
                if t is None:
                    continue
                out.append(t.detach().reshape(-1))
        return torch.cat(out)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        x = self._flat(None)
        if self._x0 is None:
            self._x0 = x.clone()
            self._rbar = float(self.param_groups[0]["reps_rel"] * (1.0 + float(x.norm())))
        g = self._flat("grad")
        self._g2 += float(g.pow(2).sum())
        self._rbar = max(self._rbar, float((x - self._x0).norm()))
        if self._g2 <= 0.0:
            return loss                      # a zero gradient everywhere: nothing to do
        eta = self._rbar / math.sqrt(self._g2)
        self.last_eta = eta
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-eta)
        return loss


class Prodigy(torch.optim.Optimizer):
    """Prodigy (D-adapted Adam), pinned: betas (0.9, 0.999), eps 1e-8, bias
    correction on, d_coef 1, no weight decay, unbounded growth."""

    def __init__(self, params, d0: float = 1e-6):
        if not d0 > 0.0:
            raise ValueError(f"d0 must be > 0, got {d0!r}")
        super().__init__(params, {"d": float(d0), "d0": float(d0), "d_max": float(d0),
                                  "d_numerator": 0.0, "k": 0})
        self.betas = (0.9, 0.999)
        self.eps = 1e-8

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        beta1, beta2 = self.betas
        beta3 = math.sqrt(beta2)
        group = self.param_groups[0]
        d, d0, d_max, k = group["d"], group["d0"], group["d_max"], group["k"]
        bias_correction = math.sqrt(1.0 - beta2 ** (k + 1)) / (1.0 - beta1 ** (k + 1))
        dlr = d * bias_correction                      # lr = 1
        d_numerator = group["d_numerator"] * beta3
        d_denom = 0.0
        for g_ in self.param_groups:
            for p in g_["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["s"] = torch.zeros_like(p)
                    state["p0"] = p.detach().clone()
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                d_numerator += (d / d0) * dlr * float(torch.dot(
                    grad.reshape(-1), (state["p0"] - p).reshape(-1)))
                state["exp_avg"].mul_(beta1).add_(grad, alpha=d * (1.0 - beta1))
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=d * d * (1.0 - beta2))
                state["s"].mul_(beta3).add_(grad, alpha=(d / d0) * dlr)
                d_denom += float(state["s"].abs().sum())
        if d_denom == 0.0:
            return loss
        d_hat = d_numerator / d_denom
        if d == d0:
            d = max(d, d_hat)
        d_max = max(d_max, d_hat)
        # reference: d = min(d_max, d * growth_rate); with growth_rate = inf
        # that is d_max. (min(d_max, d) would freeze d at d0 forever.)
        d = d_max
        for g_ in self.param_groups:
            for p in g_["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                denom = state["exp_avg_sq"].sqrt().add_(d * self.eps)
                p.addcdiv_(state["exp_avg"], denom, value=-dlr)
        group.update({"d": d, "d_max": d_max, "d_numerator": d_numerator, "k": k + 1})
        return loss


# ============================================================ Levenberg-Marquardt
def _stacked_residual(net, pde, X_col, blocks, n_eqs, params, jacobian: bool, chunk: int,
                      clear_cache):
    """The weighted residual vector R (loss = ||R||^2 = sum_i w_i mean(r_i^2))
    and, when asked, its Jacobian w.r.t. ``params`` (float64 on the device).

    ``blocks`` lists ``('pde', eq_idx, scale)`` and ``('obs', bc, scale)`` with
    ``scale = sqrt(w_i / N_i)``. The Jacobian is built a chunk of rows at a
    time with batched vector-Jacobian products; DeepXDE's derivative cache is
    cleared after each chunk (it is keyed by tensor, so reusing it across
    chunks would be wrong and would grow without bound).
    """
    R, J = [], []
    dev, dt = params[0].device, params[0].dtype

    def rows(values, keep):
        """Append ``values`` (C,) to R and, if needed, its C Jacobian rows."""
        R.append(values.detach().to(torch.float64))
        if jacobian:
            eye = torch.eye(values.numel(), dtype=values.dtype, device=values.device)
            grads = torch.autograd.grad(values, params, grad_outputs=eye,
                                        is_grads_batched=True, retain_graph=keep,
                                        allow_unused=True)
            J.append(torch.cat([(g if g is not None else torch.zeros(
                values.numel(), *p.shape, dtype=values.dtype, device=values.device))
                .reshape(values.numel(), -1) for g, p in zip(grads, params)], dim=1)
                .to(torch.float64))

    pde_blocks = [b for b in blocks if b[0] == "pde"]
    for start in range(0, X_col.shape[0], chunk):
        x = X_col[start:start + chunk].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            y = net(x)
            f = pde(x, y)
            f = f if isinstance(f, (list, tuple)) else [f]
            for i, (_, eq_idx, scale) in enumerate(pde_blocks):
                rows(f[eq_idx].reshape(-1) * scale, keep=i < len(pde_blocks) - 1)
        clear_cache()
    for kind, bc, scale in (b for b in blocks if b[0] == "obs"):
        points = torch.as_tensor(bc.points, dtype=dt, device=dev)
        values = torch.as_tensor(bc.values, dtype=dt, device=dev).reshape(-1)
        for start in range(0, points.shape[0], chunk):
            with torch.enable_grad():
                out = net(points[start:start + chunk])[:, bc.component]
                rows((out - values[start:start + chunk]) * scale, keep=False)
    R = torch.cat(R)
    return R, (torch.cat(J) if jacobian else None)


def levenberg_marquardt(net, pde, X_col, bcs, weights, n_eqs, maxiter: int = 500,
                        tau: float = 1e-3, gtol: float = 1e-8, chunk: int = 256,
                        clear_cache=lambda: None, iteration_hook=None):
    """Minimise sum_i w_i mean(r_i^2) over ``net``'s parameters by LM.

    ``weights`` are DeepXDE's loss weights in its order (one per equation, then
    one per observation block). Nielsen's damping: accept when the gain ratio
    rho > 0 and shrink lambda by max(1/3, 1 - (2 rho - 1)^3), else grow it by
    nu (doubling). The step solves (J^T J + lambda I) delta = -J^T R in float64.
    """
    params = [p for p in net.parameters() if p.requires_grad]
    dev = params[0].device
    blocks = [("pde", e, math.sqrt(weights[e] / X_col.shape[0])) for e in range(n_eqs)]
    blocks += [("obs", bc, math.sqrt(weights[n_eqs + i] / len(bc.points)))
               for i, bc in enumerate(bcs)]

    def flat():
        return torch.cat([p.detach().reshape(-1) for p in params]).to(torch.float64)

    def assign(vec):
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.copy_(vec[offset:offset + n].reshape(p.shape).to(p.dtype))
                offset += n

    t0 = time.perf_counter()
    R, J = _stacked_residual(net, pde, X_col, blocks, n_eqs, params, True, chunk, clear_cache)
    loss = float(R @ R)
    A = J.T @ J
    g = J.T @ R
    lam = tau * float(A.diagonal().max())
    nu = 2.0
    eye = torch.eye(A.shape[0], dtype=torch.float64, device=dev)
    history = [(0, loss, lam, float("nan"), time.perf_counter() - t0)]
    reason, accepted = "maxiter", 0
    for it in range(1, int(maxiter) + 1):
        if not math.isfinite(loss):
            reason = "nonfinite"
            break
        if float(g.abs().max()) <= gtol:
            reason = "gtol"
            break
        theta = flat()
        try:
            delta = torch.linalg.solve(A + lam * eye, -g)
        except RuntimeError:
            lam *= nu
            nu *= 2.0
            continue
        if float(delta.norm()) <= torch.finfo(params[0].dtype).eps * (float(theta.norm()) + 1.0):
            reason = "small_step"
            break
        assign(theta + delta)
        R_new, _ = _stacked_residual(net, pde, X_col, blocks, n_eqs, params, False, chunk,
                                     clear_cache)
        loss_new = float(R_new @ R_new)
        predicted = float(delta @ (lam * delta - g))      # = L(0) - L_model(delta), >0
        rho = (loss - loss_new) / predicted if predicted > 0 else -1.0
        if math.isfinite(loss_new) and rho > 0:
            accepted += 1
            R, J = _stacked_residual(net, pde, X_col, blocks, n_eqs, params, True, chunk,
                                     clear_cache)
            loss = float(R @ R)
            A = J.T @ J
            g = J.T @ R
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
            if iteration_hook is not None:
                iteration_hook(it, loss)
        else:
            assign(theta)
            lam *= nu
            nu *= 2.0
        history.append((it, loss, lam, rho, time.perf_counter() - t0))
    return {"reason": reason, "iterations": len(history) - 1, "accepted": accepted,
            "loss": loss, "lambda": lam, "seconds": time.perf_counter() - t0,
            "n_rows": int(R.numel()), "n_params": int(sum(p.numel() for p in params)),
            "history": history}
