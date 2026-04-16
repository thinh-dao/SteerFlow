import torch
import numpy as np
import torch.nn.functional as F

from typing import List
from tqdm import tqdm
from PIL import Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers import StableDiffusion3Pipeline, FluxPipeline, FlowMatchEulerDiscreteScheduler
from utils import (
    decode_flux_latents, decode_sd3_latents,
    calc_v_flux, calc_v_sd3_single,
    calculate_shift, visualize_mask,
)

@torch.no_grad()
def steerflow_edit(
    model_type: str,
    pipe,
    scheduler,
    z_0: torch.FloatTensor,
    src_prompt: str,
    tar_prompt: str,
    params: dict,
    external_mask=None,
    capture_trajectory: bool = False,
    capture_mask: bool = False,
):
    if model_type == "FLUX":
        # Prepare mask for FLUX packed format [B, seq_len, C]
        if external_mask is not None:
            seq_len = z_0.shape[1]
            latent_size = int(np.sqrt(seq_len))
            src_mask = torch.from_numpy(np.array(external_mask)).float() / 255.0
            src_mask = F.interpolate(
                src_mask.unsqueeze(0).unsqueeze(0),
                size=(latent_size, latent_size),
                mode="bilinear",
                align_corners=False,
            ).to(z_0.device)
        else:
            src_mask = None

        inversion_res = ode_inversion_flux(
            pipe=pipe, scheduler=scheduler, z_0=z_0,
            src_prompt=src_prompt,
            T_steps=params["T_steps"],
            forward_steps=params.get("forward_steps", 1),
            guidance_scale=params["src_guidance_scale"],
            capture_trajectory=capture_trajectory,
        )
        trajectory = inversion_res["latents_history"]

        edit_res = ode_denoise_flux(
            pipe=pipe, scheduler=scheduler,
            inverse_trajectory=trajectory,
            prompt=tar_prompt,
            ext_mask=src_mask,
            mask_params=params["mask_params"],
            T_steps=params["T_steps"],
            alpha=params["alpha"],
            guidance_scale=params["tar_guidance_scale"],
            capture_trajectory=capture_trajectory,
            capture_mask=capture_mask,
        )

    elif model_type == "SD3":
        # Prepare mask for SD3 spatial format [B, C, H, W]
        if external_mask is not None:
            _, _, lat_H, lat_W = z_0.shape
            src_mask = torch.from_numpy(np.array(external_mask)).float() / 255.0
            src_mask = F.interpolate(
                src_mask.unsqueeze(0).unsqueeze(0),
                size=(lat_H, lat_W),
                mode="bilinear",
                align_corners=False,
            ).to(z_0.device)
        else:
            src_mask = None

        negative_prompt = params.get("negative_prompt", "")

        inversion_res = ode_inversion_sd3(
            pipe=pipe, scheduler=scheduler, z_0=z_0,
            src_prompt=src_prompt,
            negative_prompt=negative_prompt,
            T_steps=params["T_steps"],
            forward_steps=params.get("forward_steps", 1),
            guidance_scale=params["src_guidance_scale"],
            capture_trajectory=capture_trajectory,
        )
        trajectory = inversion_res["latents_history"]

        edit_res = ode_denoise_sd3(
            pipe=pipe, scheduler=scheduler,
            inverse_trajectory=trajectory,
            prompt=tar_prompt,
            negative_prompt=negative_prompt,
            ext_mask=src_mask,
            mask_params=params["mask_params"],
            T_steps=params["T_steps"],
            alpha=params["alpha"],
            guidance_scale=params["tar_guidance_scale"],
            capture_trajectory=capture_trajectory,
            capture_mask=capture_mask,
        )

    return edit_res


# ==================== Masking ====================

@torch.no_grad()
def compute_velocity_mask_flux(
    V_edit: torch.Tensor,
    ext_mask,
    mask_params,
) -> torch.Tensor:
    """
    Compute a robust soft mask using Element-wise Sigmoid.
    V_edit: [B, N, C] (FLUX packed latents)
    ext_mask: [B, 1, H, W] or None
    Returns: [B, N, 1]
    """
    B, N, C = V_edit.shape
    original_dtype = V_edit.dtype
    H = W = int(np.sqrt(N))

    mask = torch.norm(V_edit.float(), dim=-1, keepdim=True).permute(0, 2, 1).view(B, 1, H, W)

    v_min = torch.quantile(mask.view(B, -1), 1 - mask_params["upper_quantile"], dim=-1).view(B, 1, 1, 1)
    v_max = torch.quantile(mask.view(B, -1), mask_params["upper_quantile"], dim=-1).view(B, 1, 1, 1)
    mask = (mask - v_min) / (v_max - v_min + 1e-8)

    x = mask_params["sigmoid_temp"] * (mask - 0.5)
    mask = torch.sigmoid(x)

    if ext_mask is not None:
        mask = torch.max(mask, ext_mask)

    if mask_params["dilation_kernel"] > 1:
        k = mask_params["dilation_kernel"]
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
        mask = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)

    return mask.view(B, 1, N).permute(0, 2, 1).to(original_dtype)


@torch.no_grad()
def compute_velocity_mask_sd3(
    V_edit: torch.Tensor,
    ext_mask,
    mask_params,
) -> torch.Tensor:
    """
    Compute a robust soft mask using Element-wise Sigmoid.
    V_edit: [B, C, H, W] (SD3 spatial latents)
    ext_mask: [B, 1, H, W] or None
    Returns: [B, 1, H, W]
    """
    B, C, H, W = V_edit.shape
    original_dtype = V_edit.dtype

    mask = torch.norm(V_edit.float(), dim=1, keepdim=True)

    v_min = torch.quantile(mask.view(B, -1), 1 - mask_params["upper_quantile"], dim=-1).view(B, 1, 1, 1)
    v_max = torch.quantile(mask.view(B, -1), mask_params["upper_quantile"], dim=-1).view(B, 1, 1, 1)
    mask = (mask - v_min) / (v_max - v_min + 1e-8)

    x = mask_params["sigmoid_temp"] * (mask - 0.5)
    mask = torch.sigmoid(x)

    if ext_mask is not None:
        mask = torch.max(mask, ext_mask)

    if mask_params.get("dilation_kernel", 1) > 1:
        k = mask_params["dilation_kernel"]
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
        mask = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)

    return mask.to(original_dtype)


# ==================== FLUX ODE Functions ====================

@torch.no_grad()
def ode_inversion_flux(
    pipe: FluxPipeline,
    scheduler: FlowMatchEulerDiscreteScheduler,
    z_0: torch.FloatTensor,
    src_prompt: str,
    T_steps: int,
    forward_steps: int,
    guidance_scale: float,
    capture_trajectory: bool,
):
    device = z_0.device
    dtype = z_0.dtype
    batch_size = z_0.shape[0]
    seq_len = z_0.shape[1]
    side = int(seq_len ** 0.5)

    latent_image_ids = pipe._prepare_latent_image_ids(batch_size, side, side, device, dtype)

    sigmas_np = np.linspace(1.0, 1.0 / T_steps, T_steps)
    mu = calculate_shift(
        seq_len, scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len, scheduler.config.base_shift, scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None, sigmas=sigmas_np, mu=mu)

    pipe._guidance_scale = guidance_scale
    src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt, prompt_2=None, device=device,
    )

    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.tensor([guidance_scale], device=device).expand(batch_size)

    z_t = z_0.clone()
    trajectory = [z_t.clone()]
    sigmas = torch.flip(scheduler.sigmas, dims=(0,))

    prev_V = None
    for i in tqdm(range(T_steps), desc="Forward ODE"):
        t_curr = sigmas[i]
        t_next = sigmas[i + 1]
        dt = t_next - t_curr

        if forward_steps > 0:
            if prev_V is not None:
                V_t = prev_V
            else:
                V_t = calc_v_flux(pipe, z_t, src_prompt_embeds, src_pooled_prompt_embeds, guidance, src_text_ids, latent_image_ids, t_curr)

            if forward_steps > 1 and i == 0:
                for _ in range(forward_steps):
                    z_t_next = z_t + dt * V_t
                    V_t = calc_v_flux(pipe, z_t_next, src_prompt_embeds, src_pooled_prompt_embeds, guidance, src_text_ids, latent_image_ids, t_curr + dt)
            else:
                z_t_next = z_t + dt * V_t
                V_t = calc_v_flux(pipe, z_t_next, src_prompt_embeds, src_pooled_prompt_embeds, guidance, src_text_ids, latent_image_ids, t_curr + dt)

            prev_V = V_t
        else:
            V_t = calc_v_flux(pipe, z_t, src_prompt_embeds, src_pooled_prompt_embeds, guidance, src_text_ids, latent_image_ids, t_curr)

        z_t = z_t + dt * V_t
        trajectory.append(z_t.clone())

    result = decode_flux_latents(z_t, pipe)[0]
    trajectory_decoded = []
    if capture_trajectory:
        trajectory_decoded = [decode_flux_latents(z.to(torch.float32), pipe) for z in trajectory]

    return {
        "images": result,
        "latents": z_t,
        "history": trajectory_decoded,
        "latents_history": trajectory,
    }


@torch.no_grad()
def ode_denoise_flux(
    pipe: FluxPipeline,
    scheduler: FlowMatchEulerDiscreteScheduler,
    inverse_trajectory: List[torch.FloatTensor],
    prompt: str,
    ext_mask,
    mask_params: dict,
    T_steps: int,
    alpha: float,
    guidance_scale: float,
    capture_trajectory: bool,
    capture_mask: bool,
    no_masked_edit: bool = False,
):
    device = pipe.transformer.device
    dtype = pipe.transformer.dtype

    B, N, C = inverse_trajectory[0].shape
    H = W = int(N ** 0.5)
    latent_image_ids = pipe._prepare_latent_image_ids(B, H, W, device, dtype)

    src_traj = inverse_trajectory[::-1]

    sigmas_np = np.linspace(1.0, 1.0 / T_steps, T_steps)
    mu = calculate_shift(
        N, scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len, scheduler.config.base_shift, scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None, sigmas=sigmas_np, mu=mu)

    pipe._guidance_scale = guidance_scale
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=device,
    )

    zt_edit = src_traj[0].clone().to(device)
    sigmas = scheduler.sigmas

    trajectory = []
    if capture_trajectory:
        trajectory.append(zt_edit.clone())

    final_mask = None

    for i in tqdm(range(T_steps), desc="Backward ODE"):
        t_curr = sigmas[i]
        t_next = sigmas[i + 1]
        dt = t_next - t_curr

        # 1. Source velocity from trajectory
        Vt_src = (src_traj[i + 1].to(device) - src_traj[i].to(device)) / dt

        # 2. Target velocity
        guidance = torch.tensor([guidance_scale], device=device).expand(B)
        Vt_tgt = calc_v_flux(pipe, zt_edit, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t_curr)

        # 3. Edit coefficient with cosine similarity
        cos_sim = F.cosine_similarity(Vt_tgt, Vt_src, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1)
        edit_coeff = (1 - t_next ** alpha) * cos_sim

        # 4. Adaptive mask (applied at every step)
        V_diff = Vt_tgt - Vt_src
        if not no_masked_edit:
            final_mask = compute_velocity_mask_flux(V_diff, ext_mask, mask_params)

        # 5. Blend velocities
        mask_term = final_mask if (final_mask is not None and not no_masked_edit) else 1.0
        blend_factor = mask_term * edit_coeff
        Vt_edit = Vt_src + blend_factor * (Vt_tgt - Vt_src)

        # 6. Euler step
        zt_edit = zt_edit + dt * Vt_edit

        if capture_trajectory:
            trajectory.append(zt_edit.clone())

    result = decode_flux_latents(zt_edit, pipe)[0]
    trajectory_decoded = []
    if capture_trajectory:
        trajectory_decoded = [decode_flux_latents(z.to(torch.float32), pipe) for z in trajectory]

    decoded_mask = None
    if not no_masked_edit and capture_mask and final_mask is not None:
        decoded_mask = visualize_mask(final_mask, size=512)

    return {
        "images": result,
        "latents": zt_edit,
        "history": trajectory_decoded,
        "mask": decoded_mask,
    }


# ==================== SD3 ODE Functions ====================

@torch.no_grad()
def ode_inversion_sd3(
    pipe: StableDiffusion3Pipeline,
    scheduler,
    z_0: torch.FloatTensor,
    src_prompt: str,
    negative_prompt: str,
    T_steps: int,
    forward_steps: int,
    guidance_scale: float,
    capture_trajectory: bool,
):
    device = z_0.device
    dtype = z_0.dtype

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._guidance_scale = guidance_scale

    (
        src_prompt_embeds, src_negative_prompt_embeds,
        src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    z_t = z_0.clone()
    trajectory = [z_t.clone()]
    sigmas = torch.flip(scheduler.sigmas, dims=(0,))

    prev_V = None
    for i in tqdm(range(T_steps), desc="Inversion (Forward ODE)"):
        t_curr = sigmas[i]
        t_next = sigmas[i + 1]
        dt = t_next - t_curr
        t_curr_scaled = t_curr * 1000
        t_next_scaled = t_next * 1000

        if forward_steps > 0:
            if prev_V is not None:
                V_t = prev_V
            else:
                V_t = calc_v_sd3_single(
                    pipe, z_t,
                    src_prompt_embeds, src_negative_prompt_embeds,
                    src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds,
                    guidance_scale, t_curr_scaled,
                )

            if forward_steps > 1 and i == 0:
                for _ in range(forward_steps):
                    z_t_next = z_t + dt * V_t
                    V_t = calc_v_sd3_single(
                        pipe, z_t_next,
                        src_prompt_embeds, src_negative_prompt_embeds,
                        src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds,
                        guidance_scale, t_next_scaled,
                    )
            else:
                z_t_next = z_t + dt * V_t
                V_t = calc_v_sd3_single(
                    pipe, z_t_next,
                    src_prompt_embeds, src_negative_prompt_embeds,
                    src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds,
                    guidance_scale, t_next_scaled,
                )

            prev_V = V_t
        else:
            V_t = calc_v_sd3_single(
                pipe, z_t,
                src_prompt_embeds, src_negative_prompt_embeds,
                src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds,
                guidance_scale, t_curr_scaled,
            )

        z_t = z_t + dt * V_t
        trajectory.append(z_t.clone())

    result = decode_sd3_latents(z_t, pipe)[0]
    trajectory_decoded = []
    if capture_trajectory:
        trajectory_decoded = [decode_sd3_latents(z.to(device), pipe) for z in trajectory]

    return {
        "images": result,
        "latents": z_t,
        "history": trajectory_decoded,
        "latents_history": trajectory,
    }


@torch.no_grad()
def ode_denoise_sd3(
    pipe: StableDiffusion3Pipeline,
    scheduler,
    inverse_trajectory: List[torch.FloatTensor],
    prompt: str,
    negative_prompt: str,
    ext_mask,
    mask_params: dict,
    T_steps: int,
    alpha: float,
    guidance_scale: float,
    capture_trajectory: bool,
    capture_mask: bool,
    no_masked_edit: bool = False,
):
    device = pipe.transformer.device
    dtype = pipe.transformer.dtype

    B, C, H, W = inverse_trajectory[0].shape
    src_traj = inverse_trajectory[::-1]

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._guidance_scale = guidance_scale

    (
        tar_prompt_embeds, tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
    )

    cfg_base_embeds = tar_negative_prompt_embeds
    cfg_base_pooled = tar_negative_pooled_prompt_embeds

    zt_edit = src_traj[0].clone().to(device=device, dtype=dtype)
    sigmas = scheduler.sigmas

    trajectory = []
    if capture_trajectory:
        trajectory.append(zt_edit.clone())

    final_mask = None

    for i in tqdm(range(T_steps), desc="Denoising (Backward ODE)"):
        t_curr = sigmas[i]
        t_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.tensor(0.0, device=device)
        dt = t_next - t_curr

        # 1. Source velocity from trajectory
        s_curr = src_traj[i].to(device=device, dtype=dtype)
        s_next = src_traj[i + 1].to(device=device, dtype=dtype)
        Vt_src = (s_next - s_curr) / dt

        # 2. Target velocity (batched CFG)
        t_scaled = t_curr * 1000
        t_tensor = t_scaled.expand(B * 2).to(dtype=dtype)
        latent_cfg = torch.cat([zt_edit, zt_edit]).to(dtype=dtype)
        prompt_cfg = torch.cat([cfg_base_embeds, tar_prompt_embeds])
        pooled_cfg = torch.cat([cfg_base_pooled, tar_pooled_prompt_embeds])

        noise_pred = pipe.transformer(
            hidden_states=latent_cfg,
            timestep=t_tensor,
            encoder_hidden_states=prompt_cfg,
            pooled_projections=pooled_cfg,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        noise_base, noise_cond = noise_pred.chunk(2)
        Vt_tgt = noise_base + guidance_scale * (noise_cond - noise_base)

        # 3. Edit coefficient with cosine similarity
        cos_sim = F.cosine_similarity(
            Vt_tgt.view(B, -1), Vt_src.view(B, -1), dim=-1
        ).view(B, 1, 1, 1)
        edit_coeff = (1 - t_next ** alpha) * cos_sim

        # 4. Adaptive mask (use conditional velocity for mask computation, applied at every step)
        V_diff = noise_cond - Vt_src
        if not no_masked_edit:
            final_mask = compute_velocity_mask_sd3(
                V_diff,
                ext_mask.to(device=device, dtype=dtype) if ext_mask is not None else None,
                mask_params,
            )

        # 5. Blend velocities
        mask_term = final_mask if (final_mask is not None and not no_masked_edit) else 1.0
        blend_factor = mask_term * edit_coeff
        Vt_edit = Vt_src + blend_factor * (Vt_tgt - Vt_src)

        # 6. Euler step
        zt_edit = zt_edit + dt * Vt_edit

        if capture_trajectory:
            trajectory.append(zt_edit.clone().to("cpu"))

    result = decode_sd3_latents(zt_edit, pipe)[0]
    trajectory_decoded = []
    if capture_trajectory:
        trajectory_decoded = [decode_sd3_latents(z.to(device), pipe)[0] for z in trajectory]

    decoded_mask = None
    if not no_masked_edit and capture_mask and final_mask is not None:
        mask_np = final_mask[0, 0].cpu().float().numpy()
        mask_np = (mask_np * 255).astype(np.uint8)
        decoded_mask = Image.fromarray(mask_np, mode='L').resize((512, 512), Image.BILINEAR)

    return {
        "images": result,
        "latents": zt_edit,
        "history": trajectory_decoded,
        "mask": decoded_mask,
    }
