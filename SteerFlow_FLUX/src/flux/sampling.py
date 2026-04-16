import math
from typing import Callable

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


@torch.no_grad()
def compute_velocity_mask(
    V_edit: torch.Tensor,
    ext_mask,
    mask_params,
) -> torch.Tensor:
    """
    Compute a soft mask from velocity difference.
    Pipeline: Magnitude -> Robust Norm -> Sigmoid -> Combine External Mask -> Morphological Closing
    """
    B, N, C = V_edit.shape
    original_dtype = V_edit.dtype
    H = W = int(np.sqrt(N))

    # Magnitude
    mask = torch.norm(V_edit.float(), dim=-1, keepdim=True).permute(0, 2, 1).view(B, 1, H, W)

    # Robust scaling via quantiles
    v_min = torch.quantile(mask.view(B, -1), 1 - mask_params["upper_quantile"], dim=-1).view(B, 1, 1, 1)
    v_max = torch.quantile(mask.view(B, -1), mask_params["upper_quantile"], dim=-1).view(B, 1, 1, 1)
    mask = (mask - v_min) / (v_max - v_min + 1e-8)

    # Sigmoid
    x = mask_params["sigmoid_temp"] * (mask - 0.5)
    mask = torch.sigmoid(x)

    # Combine with external mask
    if ext_mask is not None:
        mask = torch.max(mask, ext_mask)

    # Morphological closing
    if mask_params["dilation_kernel"] > 1:
        k = mask_params["dilation_kernel"]
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
        mask = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)

    return mask.view(B, 1, N).permute(0, 2, 1).to(original_dtype)


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15,
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def denoise(
    model: Flux,
    img: Tensor, img_ids: Tensor,
    txt: Tensor, txt_ids: Tensor, vec: Tensor,
    timesteps: list[float],
    inverse, info,
    guidance: float = 4.0,
):
    inject_list = [True] * info["inject_step"] + [False] * (len(timesteps[:-1]) - info["inject_step"])
    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info["t"] = t_prev if inverse else t_curr
        info["inverse"] = inverse
        info["second_order"] = False
        info["inject"] = inject_list[i]

        pred, info = model(
            img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
            y=vec, timesteps=t_vec, guidance=guidance_vec, info=info,
        )
        img = img + (t_prev - t_curr) * pred

    return img, info


def edit_steerflow(
    model: Flux,
    img: Tensor, img_ids: Tensor,
    txt: Tensor, txt_ids: Tensor, vec: Tensor,
    timesteps: list[float],
    inverse, info,
    guidance: float = 4.0,
    **kwargs,
):
    mask_params = info["mask_params"]
    forward_steps = info["forward_steps"]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    def call_model(img_curr, t_curr_val):
        t_vec = torch.full((img.shape[0],), t_curr_val, dtype=img.dtype, device=img.device)
        return model(
            img=img_curr, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
            y=vec, timesteps=t_vec, guidance=guidance_vec, info=info,
        )

    if inverse:
        # Forward ODE: inversion (0 -> 1)
        trajectory = [img.clone()]
        timesteps = timesteps[::-1]
        prev_V = None

        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            info.update({"t": t_prev, "inverse": inverse, "inject": False})
            dt = t_prev - t_curr

            if forward_steps > 0:
                if prev_V is not None:
                    V_t = prev_V
                else:
                    V_t, info = call_model(img, t_curr)

                if forward_steps > 1 and i == 0:
                    for _ in range(forward_steps):
                        img_next = img + dt * V_t
                        V_t, info = call_model(img_next, t_prev)
                else:
                    img_next = img + dt * V_t
                    V_t, info = call_model(img_next, t_prev)
            else:
                V_t, info = call_model(img, t_curr)

            if forward_steps > 0:
                prev_V = V_t

            img = img + dt * V_t
            trajectory.append(img.clone())

        info["trajectory"] = trajectory
        return img, info

    else:
        # Backward ODE: editing (1 -> 0)
        src_traj = info["trajectory"][::-1]
        if not src_traj:
            raise ValueError("No trajectory found for editing")

        alpha = info["alpha"]
        ext_mask = info["external_mask"]

        B, N, C = img.shape

        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            dt = t_prev - t_curr

            # Source velocity from trajectory
            Vt_src = (src_traj[i + 1].to(img.device) - src_traj[i].to(img.device)) / dt

            # Target velocity
            Vt_tgt, info = call_model(img, t_curr)

            # Edit coefficient with cosine similarity
            cos_sim = F.cosine_similarity(Vt_tgt, Vt_src, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1)
            edit_coeff = (1 - t_prev**alpha) * cos_sim

            # Adaptive mask (applied at every step)
            V_diff = Vt_tgt - Vt_src
            mask = compute_velocity_mask(V_diff, ext_mask, mask_params)

            # Blend velocities
            blend_factor = mask * edit_coeff
            Vt_edit = Vt_src + blend_factor * (Vt_tgt - Vt_src)

            img = img + dt * Vt_edit

        return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2, pw=2,
    )
