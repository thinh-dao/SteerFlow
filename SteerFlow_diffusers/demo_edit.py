import os
import argparse

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FluxPipeline

from steerflow import steerflow_edit
from utils import (
    prepare_latents_from_image_flux,
    prepare_latents_from_image_sd3,
    decode_flux_latents,
    decode_sd3_latents,
)


PARAMS = {
    "SD3": {
        "src_guidance_scale": 1.0,
        "tar_guidance_scale": 6.5,
        "T_steps": 30,
        "alpha": 5.5,
        "forward_steps": 1,
        "negative_prompt": "",
        "mask_params": {
            "sigmoid_temp": 15.0,
            "dilation_kernel": 5,
            "upper_quantile": 0.95,
        },
    },
    "FLUX": {
        "src_guidance_scale": 1.0,
        "tar_guidance_scale": 3.5,
        "T_steps": 15,
        "alpha": 4.5,
        "forward_steps": 1,
        "mask_params": {
            "sigmoid_temp": 15.0,
            "dilation_kernel": 5,
            "upper_quantile": 0.95,
        },
    },
}


def main():
    parser = argparse.ArgumentParser("SteerFlow: Image editing via controlled flow matching")
    parser.add_argument("--model_type", type=str, required=True, choices=["FLUX", "SD3"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default="")
    parser.add_argument("--src_prompt", type=str, default="")
    parser.add_argument("--tar_prompt", type=str, required=True)
    parser.add_argument("--save_folder", type=str, default="outputs")

    # Override default params
    parser.add_argument("--T_steps", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--tar_guidance_scale", type=float, default=None)

    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    device = "cuda"
    dtype = torch.float16

    # Load model
    if args.model_type == "FLUX":
        pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
        encode_fn = prepare_latents_from_image_flux
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
        encode_fn = prepare_latents_from_image_sd3

    # Load and preprocess image
    image = Image.open(args.image_path).convert("RGB")
    w, h = image.size
    w, h = w - w % 16, h - h % 16
    image = image.crop((0, 0, w, h))

    # Encode image to latents
    z_0 = encode_fn(pipe, image)

    # Load external mask
    external_mask = None
    if args.mask_path and os.path.exists(args.mask_path):
        mask_img = Image.open(args.mask_path).convert("L")
        external_mask = mask_img.resize((w, h), Image.BILINEAR)

    # Apply parameter overrides
    params = PARAMS[args.model_type]
    if args.T_steps is not None:
        params["T_steps"] = args.T_steps
    if args.alpha is not None:
        params["alpha"] = args.alpha
    if args.tar_guidance_scale is not None:
        params["tar_guidance_scale"] = args.tar_guidance_scale

    # Run editing
    result = steerflow_edit(
        model_type=args.model_type,
        pipe=pipe,
        scheduler=pipe.scheduler,
        z_0=z_0,
        src_prompt=args.src_prompt,
        tar_prompt=args.tar_prompt,
        params=params,
        external_mask=external_mask,
    )

    # Save results
    result["images"].save(os.path.join(args.save_folder, "edited.png"))
    image.save(os.path.join(args.save_folder, "input.png"))
    print(f"Results saved to {args.save_folder}")


if __name__ == "__main__":
    main()
