import torch
import math
import torch
from torch.optim import Optimizer
from collections import defaultdict
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
            lr: float = 1,
            **kwargs):
        """
        Implementation of the KOALA-V(Vanilla) optimizer

        :param params: parameters to optimize
        :param sigma: initial value of P_k
        :param q: fixed constant Q_k
        :param r: fixed constant R_k (None for online estimation)
        :param alpha_r: smoothing coefficient for online estimation of R_k
        :param weight_decay: weight decay
        :param lr: learning rate
        :param kwargs:
        """
        super(VanillaKOALA, self).__init__(params, **kwargs)

        self.eps = 1e-9

        for group in self.param_groups:
            group["lr"] = lr

        # Initialize state
        self.state["sigma"] = sigma
        self.state["q"] = q
        if r is not None:
            self.state["r"] = r
        else:
            self.state["r"] = ExpAverage(alpha_r, 1.0)
        self.state["weight_decay"] = weight_decay

    @torch.no_grad()
    def predict(self):
        self.state["sigma"] += self.state["q"]

    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor):
        if isinstance(self.state["r"], ExpAverage):
            self.state["r"].update(loss_var)
            cur_r = self.state["r"].get_avg()
        else:
            cur_r = self.state["r"]

        max_grad_entries = list()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.norm(p=2) < self.eps:
                    continue

                layer_grad = p.grad + self.state["weight_decay"] * p
                layer_grad_norm = layer_grad.norm(p=2)

                s = self.state["sigma"] * (layer_grad_norm ** 2) + cur_r

                layer_loss = loss + 0.5 * self.state["weight_decay"] * p.norm(p=2) ** 2
                scale = group["lr"] * layer_loss * self.state["sigma"] / s
                p.data.add_(-scale * p.grad)

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

"""
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
            **kwargs):
        super(KOALAPlusPlus, self).__init__(params, **kwargs)
        self.eps = 1e-9
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

        # 初始化每个参数状态（存储 vk、Hk、Sk、s）
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {}
                self.state[p]["vk"] = None
                self.state[p]["Hk"] = None
                self.state[p]["Sk"] = None
                # self.state[p]["s"] = None   # 新增：用于保存当前步的 s 标量

    @torch.no_grad()
    def predict(self):
        # 如果以后需要做“预测步”，可以在这里加，比如 Kalman 里的先验预测
        pass

    @torch.no_grad()
    def update(self, loss: torch.FloatTensor, loss_var: torch.FloatTensor):
        # 更新 r 状态
        if isinstance(self.state["r"], ExpAverage):
            self.state["r"].update(loss_var)
            cur_r = self.state["r"].get_avg()
        else:
            cur_r = self.state["r"]

        Q = self.state["q"]
        weight_decay = self.state["weight_decay"]

        for group in self.param_groups:
            lr = group["lr"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 提前检查梯度范数，避免后续计算
                grad_flat = p.grad.view(-1)
                grad_norm_sq = torch.dot(grad_flat, grad_flat)
                if grad_norm_sq < self.eps:
                    continue

                p_flat = p.view(-1)
                p_shape = p.shape
                
                # 获取上次状态
                vk_prev = self.state[p]["vk"]
                Hk_prev = self.state[p]["Hk"]
                Sk_prev = self.state[p]["Sk"]

                # 计算当前梯度 Hk（in-place 操作）
                if weight_decay != 0:
                    Hk = grad_flat.add(p_flat, alpha=weight_decay)
                else:
                    Hk = grad_flat.clone()
                
                # 初始化
                if vk_prev is None:
                    vk_prev = Hk.mul(self.state["sigma"])
                if Hk_prev is None:
                    Hk_prev = Hk
                
                # 预计算所有需要的点积（避免重复计算）
                dot_Hk_Hk_prev = torch.dot(Hk, Hk_prev)
                dot_Hk_prev_Hk_prev = torch.dot(Hk_prev, Hk_prev)
                dot_Hk_prev_vk_prev = torch.dot(Hk_prev, vk_prev)
                dot_Hk_vk_prev = torch.dot(Hk, vk_prev)
                dot_vk_prev_vk_prev = torch.dot(vk_prev, vk_prev)
                
                # 计算 Sk_prev
                if Sk_prev is None:
                    # vk_prev + Q * Hk_prev
                    temp = vk_prev + Q * Hk_prev
                    Sk_prev = torch.dot(temp, Hk_prev) + cur_r
                
                # 计算 lambda_k（复用已计算的点积）
                lambdak = (dot_Hk_vk_prev + Q * dot_Hk_Hk_prev) / Sk_prev
                
                # 计算 alpha（复用已计算的点积）
                alphak = dot_Hk_vk_prev/ (dot_Hk_prev_vk_prev + self.eps)
                
                # 计算 r_k（复用已计算的点积）
                # r_k = (dot_Hk_vk_prev * dot_Hk_prev_Hk_prev - dot_Hk_prev_vk_prev * dot_Hk_Hk_prev)
                # r_k /= (dot_Hk_prev_Hk_prev * dot_Hk_prev_Hk_prev)
                vk = (alphak - lambdak) * vk_prev + Q * (Hk - lambdak * Hk_prev)
                
                # 计算 vk（优化内存使用）
                # vk = (alpha - lambdak) * vk_prev + Q * (Hk - lambdak * Hk_prev) + r_k * Hk_prev
                # vk = vk_prev.mul(alpha - lambdak)
                # vk.add_(Hk, alpha=Q)
                # vk.add_(Hk_prev, alpha=-Q * lambdak + r_k)
                
                # 计算 s（复用已计算的点积）
                # s 是一个标量（scalar tensor）
                # s = (dot_Hk_prev_Hk_prev * dot_vk_prev_vk_prev - dot_Hk_prev_vk_prev * dot_Hk_prev_vk_prev)
                # s /= (dot_Hk_prev_Hk_prev * dot_Hk_prev_vk_prev + self.eps)  # 加 eps 防止 0 除
                
                # 更新 vk
                # vk.add_(Hk, alpha=s)
                # vk.add_(Hk_prev, alpha=-s * alpha)
                
                # 计算新的 Sk
                dot_Hk_Hk = torch.dot(Hk, Hk)
                dot_vk_Hk = torch.dot(vk, Hk)
                Sk_new = dot_vk_Hk + Q * dot_Hk_Hk + cur_r
                
                # 更新参数
                if weight_decay != 0:
                    dot_p_p = torch.dot(p_flat, p_flat)
                    layer_loss = loss + 0.5 * weight_decay * dot_p_p
                else:
                    layer_loss = loss
                
                # 计算更新量（in-place 操作）
                scale_factor = -lr * layer_loss / Sk_new
                update = vk.add(Hk, alpha=Q).mul_(scale_factor)
                p.data.add_(update.view(p_shape))
                
                # 更新状态（包含 s）
                self.state[p]["vk"] = vk
                self.state[p]["Hk"] = Hk
                self.state[p]["Sk"] = Sk_new
                # self.state[p]["s"] = s.detach()  # 存储本步的 s
"""

class KOALAPlusPlus(KOALABase):
    def __init__(
            self,
            params,
            sigma: float = 1.0,
            q: float = 1.0,
            r: float = None,
            alpha_r: float = 0.9,
            weight_decay: float = 0.0,
            lr: float = 1.0,
            momentum: float = 0.0,
            use_momentum_h: bool = True,
            is_symmetric: bool = True,
            **kwargs):
        super(KOALAPlusPlus, self).__init__(params, **kwargs)

        self.eps = 1e-9

        for group in self.param_groups:
            group["lr"] = lr

        self.state = {}
        self.state["sigma"] = sigma
        self.state["q"] = q
        self.state["weight_decay"] = weight_decay
        if r is not None:
            self.state["r"] = r
        else:
            self.state["r"] = ExpAverage(alpha_r, 1.0)
        self.state["momentum"] = momentum
        self.state["use_momentum_h"] = use_momentum_h
        self.is_symmetric = is_symmetric

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
        if isinstance(self.state["r"], ExpAverage):
            self.state["r"].update(loss_var)
            cur_r = self.state["r"].get_avg()
        else:
            cur_r = self.state["r"]

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or torch.dot(p.grad.view(-1), p.grad.view(-1)) < self.eps:
                    continue

                p_shape = p.shape
                state = self.state[p]
                vk_prev = state["vk"]
                Hk_prev = state["Hk"]
                Sk_prev = state["Sk"]

                layer_grad = p.grad.view(-1) + self.state["weight_decay"] * p.view(-1)
                if self.state["use_momentum_h"]:
                    if Hk_prev is None:
                        Hk = layer_grad
                    else:
                        prev_norm_sq = torch.dot(Hk_prev, Hk_prev)
                        rho_k = -torch.dot(layer_grad, Hk_prev) / (prev_norm_sq + self.eps)
                        Hk = layer_grad + rho_k * Hk_prev
                else:
                    Hk = layer_grad

                if vk_prev is None:
                    vk_prev = torch.zeros_like(Hk)
                if Hk_prev is None:
                    Hk_prev = torch.zeros_like(Hk)
                if Sk_prev is None:
                    Sk_prev = torch.as_tensor(cur_r, device=Hk.device, dtype=Hk.dtype)

                layer_loss = loss.to(device=Hk.device, dtype=Hk.dtype)
                layer_loss = layer_loss + 0.5 * self.state["weight_decay"] * torch.dot(p.view(-1), p.view(-1))
                Hk_norm2 = torch.dot(Hk, Hk)
                """
                Q = torch.as_tensor(cur_r, device=Hk.device, dtype=Hk.dtype) / (
                    layer_loss - Hk_norm2 + self.eps
                )
                """
                Q = self.state['q']

                vk = Q * Hk
                Sk_new = 2 * Q * Hk_norm2 + torch.as_tensor(cur_r, device=Hk.device, dtype=Hk.dtype)
                scale = -group["lr"] * layer_loss * (vk + Q * Hk) / Sk_new
                p.data.add_(scale.view(p_shape))

                state["vk"] = vk
                state["Hk"] = Hk
                state["Sk"] = Sk_new
