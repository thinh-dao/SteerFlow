import numpy as np
import torch
from typing import List
from PIL import Image
from contextlib import nullcontext
from diffusers import FluxPipeline


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
        t = torch.tensor([t], device=latents.device)
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    return noise_pred


def calc_v_sd3_single(pipe, latent, prompt_embeds, negative_prompt_embeds,
                      pooled_prompt_embeds, negative_pooled_prompt_embeds,
                      guidance_scale, t):
    timestep = t.expand(latent.shape[0])

    with torch.no_grad():
        if pipe.do_classifier_free_guidance:
            latent_input = torch.cat([latent, latent])
            prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
            timestep_input = timestep.repeat(2)
        else:
            latent_input = latent
            prompt_embeds_input = prompt_embeds
            pooled_embeds_input = pooled_prompt_embeds
            timestep_input = timestep

        noise_pred = pipe.transformer(
            hidden_states=latent_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds_input,
            pooled_projections=pooled_embeds_input,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return noise_pred


def get_autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "mps"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def prepare_latents_from_image_flux(pipe: FluxPipeline, image: Image.Image) -> torch.FloatTensor:
    processed = pipe.image_processor.preprocess(image)
    processed = processed.to(pipe.vae.device, dtype=pipe.vae.dtype)
    autocast_ctx = get_autocast_context(pipe.vae.device, pipe.vae.dtype)
    with autocast_ctx, torch.inference_mode():
        encoded = pipe.vae.encode(processed).latent_dist.mode()
    latents = (encoded - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4
    packed_latents = pipe._pack_latents(
        latents, latents.shape[0], num_channels_latents, latents.shape[2], latents.shape[3],
    )
    return packed_latents.to(pipe.transformer.device)


def prepare_latents_from_image_sd3(pipe, image: Image.Image) -> torch.FloatTensor:
    processed = pipe.image_processor.preprocess(image)
    processed = processed.to(pipe.vae.device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(processed).latent_dist.mode()
    latents = latents * pipe.vae.config.scaling_factor
    return latents.to(pipe.transformer.device)


def decode_flux_latents(packed: torch.FloatTensor, pipe: FluxPipeline) -> List[Image.Image]:
    num_channels_latents = pipe.transformer.config.in_channels // 4
    batch, seq_len, _ = packed.shape
    side = int(seq_len**0.5)

    latents = packed.view(batch, side, side, num_channels_latents, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5).contiguous()
    latents = latents.view(batch, num_channels_latents, side * 2, side * 2)

    to_decode = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    to_decode = to_decode.to(pipe.vae.device, dtype=pipe.vae.dtype)
    autocast_ctx = get_autocast_context(pipe.vae.device, pipe.vae.dtype)
    with autocast_ctx, torch.inference_mode():
        decoded = pipe.vae.decode(to_decode, return_dict=False)[0]
    decoded = decoded.float()
    images = pipe.image_processor.postprocess(decoded)
    return [img.copy() for img in images]


def decode_sd3_latents(latents: torch.FloatTensor, pipe) -> List[Image.Image]:
    to_decode = latents / pipe.vae.config.scaling_factor
    to_decode = to_decode.to(pipe.vae.device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(to_decode, return_dict=False)[0]
    decoded = decoded.float()
    images = pipe.image_processor.postprocess(decoded)
    return [img.copy() for img in images]


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def visualize_mask(mask, size=512):
    if mask.dim() == 3:
        mask = mask[0]
    mask = mask.squeeze(-1)
    side = int(mask.shape[0] ** 0.5)
    mask_2d = mask.view(side, side).cpu().float().numpy()
    mask_2d = (mask_2d * 255).astype(np.uint8)
    return Image.fromarray(mask_2d, mode="L").resize((size, size), Image.BILINEAR)
