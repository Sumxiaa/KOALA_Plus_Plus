import torch
import math
from util import ExpAverage


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
                vk = (alpha_k - lambda_k) * vk_prev + Q * (Hk - lambda_k * Hk_prev) + r_k * Hk_prev
                Sk_new = torch.dot(vk.add(Hk.mul(Q)), Hk).add_(cur_r)

                # Calculate the layer_loss, with the weight decay
                layer_loss = loss + 0.5 * self.state["weight_decay"] * torch.dot(p.view(-1), p.view(-1))
                # scale = - lr * layer_loss * Pk_hat * Hk / Sk_new
                scale = - group["lr"] * layer_loss * (vk + Q * Hk) / Sk_new
                p.data.add_(scale.view(p_shape))
                self.state[p]["vk"] = vk
                self.state[p]["Hk"] = Hk
                self.state[p]["Sk"] = Sk_new


