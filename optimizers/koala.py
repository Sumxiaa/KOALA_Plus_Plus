import torch

import math
import torch


class ExpAverage(object):
    def __init__(self, alpha, init_val=0):
        self.val = init_val
        self.avg = init_val
        self.alpha = alpha

    def update(self, val):
        self.val = val
        self.avg = self.alpha * self.avg + (1 - self.alpha) * val

    def get_avg(self):
        return self.avg

    def get_last_val(self):
        return self.val

"""
class KOALAPBase(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        defaults = dict(**kwargs)
        super(KOALAPBase, self).__init__(params, defaults)

    @torch.no_grad()
    def predict(self):
        pass

    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor, apply_param_update: bool = True):
        pass
"""


def _compile_if_available(fn):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(fn)
        except Exception:
            return fn
    return fn


def _zeropower_via_newtonschulz5_impl(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    X = X / (X.norm() + eps)

    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * (A @ B)

    if transposed:
        X = X.T
    return X


zeropower_via_newtonschulz5 = _compile_if_available(_zeropower_via_newtonschulz5_impl)


def _polar_factor_fast(M: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert M.ndim == 2
    if not M.is_cuda:
        return M
    return zeropower_via_newtonschulz5(M, steps=steps, eps=eps)


def _plugin_returns_full_update(opt_state) -> bool:
    return opt_state.get("gradient_plugin") in {"adamw", "muon", "sgdm"}


def _plugin_uses_strict_decoupled_weight_decay(opt_state) -> bool:
    return opt_state.get("gradient_plugin") in {"adamw", "muon"}


def _compute_plugin_tensor(opt_state, param_state, p, raw_grad, lr, wd, decoupled_wd):
    plugin = opt_state["gradient_plugin"]
    if plugin == "sgdm":
        g = raw_grad
        # Match torch.optim.SGD momentum weight decay behavior.
        if wd != 0.0:
            g = g.add(p, alpha=wd)
        buf = param_state.get("sgdm_momentum_buffer")
        if buf is None:
            buf = torch.zeros_like(g)
        momentum = opt_state.get("sgdm_momentum", 0.9)
        buf.mul_(momentum).add_(g)

        param_state["sgdm_momentum_buffer"] = buf
        return buf.mul(-lr)

    if plugin == "muon":
        g = raw_grad
        if g.ndim >= 2:
            Gmat = g.view(g.size(0), -1)
        else:
            Gmat = g.view(1, -1)

        buf = param_state.get("muon_momentum_buffer")
        if buf is None:
            # Backward compatibility for checkpoints created before the buffer rename.
            buf = param_state.get("muon_momentum")
        if buf is None:
            buf = torch.zeros_like(Gmat, dtype=torch.float32)
        buf.mul_(opt_state["muon_beta"]).add_(Gmat.float())

        if opt_state["muon_nesterov"]:
            H_in = Gmat.float().add(buf, alpha=opt_state["muon_beta"])
        else:
            H_in = buf

        if (g.ndim < 2) and opt_state["skip_polar_for_1d"]:
            Hmat = H_in
        else:
            m, n = int(H_in.size(0)), int(H_in.size(1))
            if min(m, n) < opt_state["min_polar_dim"]:
                Hmat = H_in
            else:
                Hmat = _polar_factor_fast(H_in, steps=opt_state["backend_steps"]).to(dtype=H_in.dtype)

        if opt_state["muon_scale_by_dim"]:
            m, n = int(Hmat.size(0)), int(Hmat.size(1))
            shape_scale = math.sqrt(max(1.0, float(m) / float(n)))
            scale_mul = opt_state["muon_scale"] * opt_state["muon_dim_scale_coef"] * shape_scale
        else:
            scale_mul = opt_state["muon_scale"] * opt_state["muon_dim_scale_coef"]
        if scale_mul != 1.0:
            Hmat = Hmat.mul(scale_mul)

        param_state["muon_momentum_buffer"] = buf
        update = Hmat.reshape_as(Gmat).reshape_as(raw_grad).to(dtype=raw_grad.dtype).mul(-lr)
        if decoupled_wd and wd != 0.0:
            update = update.add(p, alpha=(-lr * wd))
        return update

    if plugin == "adamw":
        g = raw_grad
        if param_state.get("adamw_exp_avg") is None:
            param_state["adamw_step"] = 0
            param_state["adamw_exp_avg"] = torch.zeros_like(g)
            param_state["adamw_exp_avg_sq"] = torch.zeros_like(g)
        exp_avg = param_state["adamw_exp_avg"]
        exp_avg_sq = param_state["adamw_exp_avg_sq"]
        beta1 = opt_state["adamw_beta1"]
        beta2 = opt_state["adamw_beta2"]
        param_state["adamw_step"] += 1
        step = param_state["adamw_step"]
        exp_avg.mul_(beta1).add_(g, alpha=(1.0 - beta1))
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2))
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        adamw_direction = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt().add_(opt_state["adamw_eps"]))
        update = adamw_direction.mul(-lr)
        if decoupled_wd and wd != 0.0:
            update = update.add(p, alpha=(-lr * wd))
        return update

    if plugin == "none":
        if decoupled_wd or wd == 0.0:
            return raw_grad
        return raw_grad.add(p, alpha=wd)

    raise ValueError(f"Unsupported gradient_plugin={plugin}")


class ExpAverage(object):
    def __init__(self, alpha, init_val=0):
        self.val = init_val
        self.avg = init_val
        self.alpha = alpha

    def update(self, val):
        self.val = val
        self.avg = self.alpha * self.avg + (1 - self.alpha) * val

    def get_avg(self):
        return self.avg

    def get_last_val(self):
        return self.val
    
    
class KOALABase(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        defaults = dict(**kwargs)
        super(KOALABase, self).__init__(params, defaults)

    @torch.no_grad()
    def predict(self):
        pass

    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor):
        pass


class VanillaKOALA(KOALABase):
    def __init__(
            self,
            params,
            sigma: float = 1,
            q: float = 1,
            r: float = None,
            alpha_r: float = 0.9,
            weight_decay: float = 0.0,
            decoupled_weight_decay: bool = False,
            lr: float = 1,
            gradient_plugin: str = "none",
            sgdm_momentum: float = 0.9,
            adamw_beta1: float = 0.9,
            adamw_beta2: float = 0.95,
            adamw_eps: float = 1e-8,
            muon_beta: float = 0.95,
            muon_nesterov: bool = True,
            backend_steps: int = 5,
            muon_scale: float = 1.0,
            muon_scale_by_dim: bool = True,
            muon_dim_scale_coef: float = 1.0,
            min_polar_dim: int = 64,
            skip_polar_for_1d: bool = True,
            use_ema_loss: bool = True,
            loss_ema_rho: float = 0.9,
            **kwargs):
        super(VanillaKOALA, self).__init__(params, **kwargs)

        self.eps = 1e-9

        for group in self.param_groups:
            group["lr"] = lr

        self.state["sigma"] = sigma
        self.state["q"] = q
        if r is not None:
            self.state["r"] = r
        else:
            self.state["r"] = ExpAverage(alpha_r, 1.0)
        self.state["weight_decay"] = weight_decay
        self.state["decoupled_weight_decay"] = bool(decoupled_weight_decay)
        self.state["gradient_plugin"] = str(gradient_plugin)
        self.state["sgdm_momentum"] = float(sgdm_momentum)
        self.state["adamw_beta1"] = float(adamw_beta1)
        self.state["adamw_beta2"] = float(adamw_beta2)
        self.state["adamw_eps"] = float(adamw_eps)
        self.state["muon_beta"] = float(muon_beta)
        self.state["muon_nesterov"] = bool(muon_nesterov)
        self.state["backend_steps"] = int(backend_steps)
        self.state["muon_scale"] = float(muon_scale)
        self.state["muon_scale_by_dim"] = bool(muon_scale_by_dim)
        self.state["muon_dim_scale_coef"] = float(muon_dim_scale_coef)
        self.state["min_polar_dim"] = int(min_polar_dim)
        self.state["skip_polar_for_1d"] = bool(skip_polar_for_1d)
        self.state["use_ema_loss"] = bool(use_ema_loss)
        self.state["loss_ema_rho"] = float(loss_ema_rho)
        self.state["ema_loss"] = None

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {}
                self.state[p]["sgdm_momentum_buffer"] = None
                self.state[p]["muon_momentum_buffer"] = None
                self.state[p]["muon_momentum"] = None
                self.state[p]["adamw_step"] = 0
                self.state[p]["adamw_exp_avg"] = None
                self.state[p]["adamw_exp_avg_sq"] = None

    @torch.no_grad()
    def predict(self):
        self.state["sigma"] += self.state["q"]

    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor, apply_param_update: bool = True):
        if isinstance(self.state["r"], ExpAverage):
            self.state["r"].update(loss_var)
            cur_r = self.state["r"].get_avg()
        else:
            cur_r = self.state["r"]
        if self.state["use_ema_loss"]:
            rho = self.state["loss_ema_rho"]
            loss_detached = loss.detach()
            if self.state["ema_loss"] is None:
                self.state["ema_loss"] = loss_detached.clone()
            else:
                self.state["ema_loss"].mul_(rho).add_(loss_detached, alpha=1.0 - rho)
            koala_loss = self.state["ema_loss"]
        else:
            koala_loss = loss

        max_grad_entries = []
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None or p.grad.norm(p=2) < self.eps:
                    continue

                wd = self.state["weight_decay"]
                decoupled_wd = self.state["decoupled_weight_decay"] or _plugin_uses_strict_decoupled_weight_decay(self.state)
                raw_grad = p.grad
                plugin_returns_update = _plugin_returns_full_update(self.state)

                if apply_param_update and (not plugin_returns_update) and decoupled_wd and wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                layer_grad = _compute_plugin_tensor(self.state, self.state[p], p, raw_grad, lr, wd, decoupled_wd)
                layer_grad_norm = layer_grad.norm(p=2)
                s = self.state["sigma"] * (layer_grad_norm ** 2) + cur_r
                
                """
                if decoupled_wd or wd == 0.0:
                    layer_loss = loss
                else:
                    layer_loss = loss + 0.5 * wd * p.norm(p=2) ** 2
                """
                
                layer_loss = koala_loss + 0.5 * wd * p.norm(p=2) ** 2

                if plugin_returns_update:
                    scale = layer_loss * self.state["sigma"] / s
                    if apply_param_update:
                        p.data.add_(layer_grad, alpha=scale)
                else:
                    scale = lr * layer_loss * self.state["sigma"] / s
                    if apply_param_update:
                        p.data.add_(layer_grad, alpha=-scale)

                max_grad_entries.append(layer_grad_norm ** 2 / s)

        hh_approx = torch.max(torch.stack(max_grad_entries))
        self.state["sigma"] -= self.state["sigma"] ** 2 * hh_approx


class MomentumKOALA(KOALABase):
    def __init__(
            self,
            params,
            sw: float = 1e-1,
            sc: float = 0,
            sv: float = 1e-1,
            a: float = 0.9,
            qw: float = 1e-2,
            qv: float = 1e-2,
            r: float = None,
            alpha_r: float = 0.9,
            weight_decay: float = 0.0,
            lr: float = 1,
            **kwargs):
        """
        Implementation of the KOALA-M(Momentum) optimizer

        :param params: parameters to optimize
        :param sw: initial value of P_k for states
        :param sc: initial value of out of diagonal entries of P_k
        :param sv: initial value of P_k for velocities
        :param a: decay coefficient for velocities
        :param qw: fixed constant Q_k for states
        :param qv: fixed constant Q_k for velocities
        :param r: fixed constant R_k (None for online estimation)
        :param alpha_r: smoothing coefficient for online estimation of R_k
        :param weight_decay: weight decay
        :param lr: learning rate
        :param kwargs:
        """
        super(MomentumKOALA, self).__init__(params, **kwargs)

        self.eps = 1e-9

        self.shared_device = self.param_groups[0]["params"][0].device
        self.dtype = torch.double

        # Initialize velocities and count params
        self.total_params = 0
        for group in self.param_groups:
            group["lr"] = lr
            for p in group["params"]:
                self.state[p]["vt"] = p.new_zeros(p.shape)
                self.state[p]["gt"] = p.new_zeros(p.shape)
                self.total_params += torch.prod(torch.Tensor(list(p.size())).to(self.shared_device))

        # Define state
        self.state["Pt"] = torch.Tensor([
            [sw, sc],
            [sc, sv]
        ]).to(self.shared_device).to(self.dtype)

        self.state["qw"] = ExpAverage(0.9, qw)
        self.state["qv"] = qv
        self.state["Q"] = torch.diag(
            torch.Tensor([self.state["qw"].get_avg(), self.state["qv"]])
        ).to(self.shared_device).to(self.dtype)

        if r is not None:
            self.state["R"] = r
        else:
            self.state["R"] = ExpAverage(alpha_r, 1.0)

        f = [[1, 1], [0, a]]
        self.state["F"] = torch.Tensor(f).to(self.shared_device).to(self.dtype)

        self.state["weight_decay"] = weight_decay

    @torch.no_grad()
    def predict(self):
        wdiff = list()
        for group in self.param_groups:
            for p in group["params"]:
                pw_diff = (self.state[p]["gt"] - p).norm(p=2).to(self.shared_device)
                wdiff.append(pw_diff)

                p.mul_(self.state["F"][0, 0].to(p.device))
                p.add_(self.state[p]["vt"] * self.state["F"][0, 1].to(p.device))
                self.state[p]["vt"].mul_(self.state["F"][1, 1].to(p.device))
                self.state[p]["vt"].add_(p * self.state["F"][1, 0].to(p.device))

        norm_wdiff = torch.stack(wdiff).norm(p=2) / self.total_params
        self.state["qw"].update(norm_wdiff)
        self.state["Q"] = torch.diag(
            torch.Tensor([self.state["qw"].get_avg(), self.state["qv"]])
        ).to(self.shared_device).to(self.dtype)

        self.state["Pt"] = torch.matmul(
            torch.matmul(self.state["F"], self.state["Pt"]), self.state["F"].t())
        self.state["Pt"].add_(self.state["Q"])

    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor):
        if isinstance(self.state["R"], ExpAverage):
            self.state["R"].update(loss_var.to(self.shared_device))
            cur_r = self.state["R"].get_avg()
        else:
            cur_r = self.state["R"]

        max_grad_entries = list()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.norm(p=2) < self.eps:
                    continue

                layer_grad = p.grad + self.state["weight_decay"] * p
                layer_grad_norm = layer_grad.norm(p=2)

                S = layer_grad_norm ** 2 * self.state["Pt"][0, 0] + cur_r

                layer_loss = loss.to(self.shared_device) + 0.5 * self.state["weight_decay"] * p.norm(p=2) ** 2
                K1 = self.state["Pt"][0, 0] / S * layer_loss * group["lr"]
                K2 = self.state["Pt"][1, 0] / S * layer_loss * group["lr"]

                # Update weights and velocities
                p.sub_((K1 * layer_grad).to(p.device))
                self.state[p]["vt"].sub_((K2 * layer_grad).to(p.device))

                self.state[p]["gt"].mul_(0.9)
                self.state[p]["gt"].add_(0.1 * p)

                max_grad_entries.append(layer_grad_norm ** 2 / S)

        hh_approx = torch.max(torch.stack(max_grad_entries))

        # Update covariance
        HHS = torch.Tensor([
            [hh_approx, 0],
            [0, 0]
        ]).to(self.shared_device).to(self.dtype)
        PHHS = torch.matmul(self.state["Pt"], HHS)
        PHHSP = torch.matmul(PHHS, self.state["Pt"].t())
        self.state["Pt"] = self.state["Pt"] - PHHSP



class KOALAPlusPlus(KOALABase):
    def __init__(
            self,
            params,
            sigma: float = 1,
            q: float = 1,
            r: float = None,
            alpha_r: float = 0.9,
            weight_decay: float = 0.0,
            lr: float = 1,
            is_symmetric: bool = True,
            **kwargs):
        super(KOALAPlusPlus, self).__init__(params, **kwargs)
        
        # 优化点：预先计算 eps 的平方，用于 update 中的 L2 范数比较，避免开方
        self.eps = 1e-9
        self.eps_sq = self.eps**2 
        
        for group in self.param_groups:
            group["lr"] = lr

        # 初始化状态（常量以数值形式存储）
        self.state = {}
        self.state["sigma"] = sigma  # σ_0
        self.state["q"] = q          # Q
        if r is not None:
            self.state["r"] = r
        else:
            self.state["r"] = ExpAverage(alpha_r, 1.0)
        self.state["weight_decay"] = weight_decay

        self.is_symmetric = is_symmetric

        # 初始化每个参数状态（存储 vk、Hk、Sk、Pk）
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {}
                self.state[p]["vk"] = None
                self.state[p]["Hk"] = None
                self.state[p]["Sk"] = None

    @torch.no_grad()
    def predict(self):
        pass



    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor):
        # update for r
        if isinstance(self.state["r"], ExpAverage):
            self.state["r"].update(loss_var)
            cur_r = self.state["r"].get_avg()
        else:
            cur_r = self.state["r"]

        is_symmetric = self.is_symmetric

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.norm(2) < self.eps:
                    continue

                p_shape = p.shape
                # get the previous state
                vk_prev = self.state[p]["vk"]
                Hk_prev = self.state[p]["Hk"]
                Sk_prev = self.state[p]["Sk"]

                Q = self.state["q"]
                sigma = self.state["sigma"]

                # 计算当前梯度 Hk（拉平成向量），使用 in-place 加法
                Hk = p.grad.view(-1) + self.state["weight_decay"] * p.view(-1)
                # 初始化 vk_prev、Hk_prev 如不存在
                if vk_prev is None:
                    vk_prev = Hk.mul(sigma)
                if Hk_prev is None:
                    Hk_prev = Hk
                # Hk_prev_vk_prev = torch.dot(Hk_prev, vk_prev)
                x = torch.dot(Hk_prev, Hk_prev)
                y = torch.dot(Hk_prev, vk_prev)
                z = torch.dot(vk_prev, vk_prev)
                # Hk_prev_norm = torch.dot(Hk_prev, Hk_prev)
                if Sk_prev is None:
                    # Sk_prev = torch.dot(vk_prev.add(Q, alpha=1.0).mul_(Hk_prev), Hk_prev).add_(cur_r)
                    Sk_prev = y + Q * x + cur_r

                # Compute lambda_k
                Hk_Hk_prev = torch.dot(Hk, Hk_prev)
                Hk_vk_prev = torch.dot(Hk, vk_prev)
                lambda_k = (Hk_vk_prev + Q * Hk_Hk_prev) / Sk_prev
                # lambdak = torch.dot(Hk, vk_prev + Q * Hk_prev) / Sk_prev
                alpha_k = Hk_Hk_prev / x
                if is_symmetric:
                    # r_k = torch.dot(Hk, vk_prev) * Hk_prev_norm - torch.dot(Hk_prev, vk_prev) * torch.dot(Hk, Hk_prev)
                    r_k = Hk_vk_prev / x - Hk_Hk_prev * y / (x**2)                                  
                else:
                    r_k = 0
                s = (x * z - y * y) / (x * y)
                vk = (alpha_k - lambda_k) * vk_prev + Q * (Hk - lambda_k * Hk_prev) + r_k * Hk_prev + s * (Hk - alpha_k * Hk_prev)
                Sk_new = torch.dot(vk, Hk) + Q * torch.dot(Hk, Hk) + cur_r

                # Calculate the layer_loss, with the weight decay
                layer_loss = loss + 0.5 * self.state["weight_decay"] * torch.dot(p.view(-1), p.view(-1))
                # scale = - lr * layer_loss * Pk_hat * Hk / Sk_new
                scale = - group["lr"] * layer_loss * (vk + Q * Hk) / Sk_new
                p.data.add_(scale.view(p_shape))
                self.state[p]["vk"] = vk
                self.state[p]["Hk"] = Hk
                self.state[p]["Sk"] = Sk_new