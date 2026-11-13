from __future__ import annotations

import math

import torch
import torch.nn as nn

#lora and oplora as one small module, so lora-16 and oplora-16 run exactly the same code (same init, dropout, scaling,
#targets) and differ only in the projection of the two factors.
#
#  lora   -> W = W0 + (alpha / r) B A,                         A in R^{r x d_in} random, B in R^{d_out x r} zero
#  oplora -> W = W0 + (alpha / r) B~ A~,   B~ = B - U_k (U_k^T B),   A~ = A - (A V_k) V_k^T
#
#U_k and V_k are the k leading left/right singular vectors of the frozen starting weight W0 = U S V^T. projecting B
#removes every component of the correction along W0's dominant output directions, projecting A every component
#along its dominant input directions, so U_k^T dW = 0 and dW V_k = 0 by construction


class AdapterLinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, protect_k: int | None = None):
        super().__init__()
        self.base = base #the original frozen linear layer (fp16 backbone weight, no bias in qwen3)
        self.r, self.scale = r, alpha / r
        d_out, d_in = base.weight.shape
        #fp32 trainable factors. A gets the same kaiming-uniform init nn.Linear (and peft) use, B starts at zero, so
        #the correction BA is exactly 0 at step 0 and every run starts from the unchanged model
        self.A = nn.Parameter(torch.empty(r, d_in, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        #protect_k None -> plain lora. otherwise the frozen fp32 bases, filled in by set_bases (computed once)
        self.protect_k = protect_k
        if protect_k is not None:
            self.register_buffer("U_k", torch.zeros(d_out, protect_k, dtype=torch.float32, device=base.weight.device))
            self.register_buffer("V_k", torch.zeros(d_in, protect_k, dtype=torch.float32, device=base.weight.device))

    def set_bases(self, U_k, V_k):
        self.U_k.copy_(U_k)
        self.V_k.copy_(V_k)

    def effective_factors(self):
        #the factors the forward pass actually uses. for oplora these are the projected B~ and A~, formed from the
        #thin bases (d x k) instead of dense d x d projection matrices -> U_k (U_k^T B) costs O(d k r), not O(d^2).
        #autocast is switched off so the projection runs in fp32 and the constraint holds to fp32 precision
        if self.protect_k is None:
            return self.B, self.A
        with torch.autocast(device_type=self.B.device.type, enabled=False):
            B = self.B - self.U_k @ (self.U_k.T @ self.B)
            A = self.A - (self.A @ self.V_k) @ self.V_k.T
        return B, A

    def delta_weight(self):
        #dense correction dW = (alpha / r) B A in fp32, for merging, export and the weight-space diagnostics
        B, A = self.effective_factors()
        return self.scale * (B.float() @ A.float())

    def forward(self, x):
        #frozen path + low rank path. dropout acts on the adapter's input only, the frozen path sees x unchanged.
        #(x A^T) B^T never builds the d_out x d_in correction, just two thin matmuls through the r dim bottleneck
        B, A = self.effective_factors()
        return self.base(x) + self.scale * ((self.dropout(x) @ A.T) @ B.T)


def target_linears(model, targets):
    #every targeted projection in every transformer layer, in a fixed (module registration) order ->
    #layer 0 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, then layer 1 ...
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name.split(".")[-1] in targets]


def inject_adapters(model, targets, r, alpha, dropout, protect_k=None):
    #freezes the whole backbone, then swaps each target nn.Linear for an AdapterLinear wrapping it. the caller seeds
    #the rng right before this, so lora-16 and oplora-16 draw the identical initial A factors (same order, same shapes)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = {}
    for name, linear in target_linears(model, targets):
        parent_name, child = name.rsplit(".", 1)
        adapter = AdapterLinear(linear, r, alpha, dropout, protect_k)
        setattr(model.get_submodule(parent_name), child, adapter)
        wrapped[name] = adapter
    return wrapped


@torch.no_grad()
def compute_bases(adapters, k):
    #one fp32 svd per frozen starting weight, done once before training. W0 = U S V^T with singular values sorted in
    #descending order, so the first k columns of U and the first k rows of V^T are the k leading directions
    bases = {}
    for name, adapter in adapters.items():
        U, S, Vh = torch.linalg.svd(adapter.base.weight.float(), full_matrices=False)
        bases[name] = (U[:, :k].contiguous(), Vh[:k].T.contiguous())
    return bases


def adapter_state(adapters):
    #the trainable state only (unprojected A and B) -> what a recovery checkpoint needs. the bases are saved separately once
    return {f"{name}.{part}": getattr(a, part).detach().cpu() for name, a in adapters.items() for part in ("A", "B")}


def load_adapter_state(adapters, state):
    for name, a in adapters.items():
        a.A.data.copy_(state[f"{name}.A"])
        a.B.data.copy_(state[f"{name}.B"])


@torch.no_grad()
def export_factors(adapters):
    #the inference export -> the effective (projected, for oplora) factors and the scale, in fp32. merging and the
    #diagnostics use these, never the unprojected training factors
    out = {}
    for name, a in adapters.items():
        B, A = a.effective_factors()
        out[f"{name}.B"], out[f"{name}.A"] = B.detach().cpu().contiguous(), A.detach().cpu().contiguous()
    return out


@torch.no_grad()
def projection_residuals(adapters):
    #numerical check of the oplora constraint on the exported factors -> relative size of what is left inside the
    #protected subspaces. both are ~1e-7 (fp32 round-off) when the projection is right, ~1 if it is not applied
    worst_out, worst_in = 0.0, 0.0
    for a in adapters.values():
        B, A = a.effective_factors()
        worst_out = max(worst_out, float((a.U_k.T @ B).norm() / B.norm().clamp_min(1e-12)))
        worst_in = max(worst_in, float((A @ a.V_k).norm() / A.norm().clamp_min(1e-12)))
    return {"max_rel_Uk_T_B": worst_out, "max_rel_A_Vk": worst_in}
