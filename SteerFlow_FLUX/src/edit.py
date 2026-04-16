import os
import re
import time
import argparse
from dataclasses import dataclass
from glob import iglob

import torch
import numpy as np
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline

from flux.sampling import edit_steerflow, denoise, get_schedule, prepare, unpack
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85


@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0).to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


@torch.inference_mode()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    name = args.name

    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        raise ValueError(f"Unknown model: {name}, choose from {', '.join(configs.keys())}")

    torch_device = torch.device(device)
    num_steps = args.num_steps

    # Load components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if args.offload else torch_device)
    ae = load_ae(name, device="cpu" if args.offload else torch_device)

    if args.offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)

    # Load image
    init_image_array = np.array(Image.open(args.source_img_dir).convert("RGB"))
    shape = init_image_array.shape
    new_h = shape[0] - shape[0] % 16
    new_w = shape[1] - shape[1] % 16
    init_image = init_image_array[:new_h, :new_w, :]
    width, height = init_image.shape[0], init_image.shape[1]

    t0 = time.perf_counter()
    init_image = encode(init_image, torch_device, ae)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        width=width, height=height,
        num_steps=num_steps,
        guidance=args.guidance,
        seed=args.seed if args.seed > 0 else None,
    )

    if opts.seed is None:
        opts.seed = rng.seed()
    print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")

    if args.offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        t5, clip = t5.to(torch_device), clip.to(torch_device)

    # Build info dict
    info = {
        "feature_path": args.feature_path,
        "feature": {},
        "inject_step": 0,
        "start_layer_index": 0,
        "end_layer_index": 0,
        "reuse_v": 0,
        "editing_strategy": "replace_v",
        "qkv_ratio": [1.0, 1.0, 1.0],
        "alpha": args.alpha,
        "mask_params": {
            "upper_quantile": args.upper_quantile,
            "sigmoid_temp": args.sigmoid_temp,
            "dilation_kernel": args.dilation_kernel,
        },
        "forward_steps": args.forward_steps,
    }

    # Load external mask
    if args.mask_path and os.path.exists(args.mask_path):
        import torch.nn.functional as F
        mask_img = Image.open(args.mask_path).resize((opts.width, opts.height))
        mask_tensor = torch.from_numpy(np.array(mask_img)).float() / 255.0
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.mean(dim=-1, keepdim=True)
        elif mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(-1)
        latent_h = opts.height // 16
        latent_w = opts.width // 16
        mask_tensor = mask_tensor.permute(2, 0, 1).unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
        info["external_mask"] = mask_tensor.to(torch_device)
    else:
        info["external_mask"] = None

    os.makedirs(args.feature_path, exist_ok=True)

    # Prepare inputs
    inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
    inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
    timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

    if args.offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    # Inversion
    z, info = edit_steerflow(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
    inp_target["img"] = z

    timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

    # Editing
    x, _ = edit_steerflow(model, **inp_target, timesteps=timesteps, guidance=args.guidance, inverse=False, info=info)

    if args.offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    # Decode and save
    batch_x = unpack(x.float(), opts.width, opts.height)
    for x in batch_x:
        x = x.unsqueeze(0)
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        output_name = os.path.join(output_dir, f"{args.output_prefix}_img_{{idx}}.jpg")
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1 if fns else 0

        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")

        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        nsfw_score = [s["score"] for s in nsfw_classifier(img) if s["label"] == "nsfw"][0]

        if nsfw_score < NSFW_THRESHOLD:
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
            exif_data[ExifTags.Base.Make] = "Black Forest Labs"
            exif_data[ExifTags.Base.Model] = name
            exif_data[ExifTags.Base.ImageDescription] = args.source_prompt
            img.save(fn, exif=exif_data, quality=95, subsampling=0)
        else:
            print("Your generated image may contain NSFW content.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SteerFlow FLUX Editing")

    parser.add_argument("--name", default="flux-dev", type=str)
    parser.add_argument("--source_img_dir", required=True, type=str)
    parser.add_argument("--source_prompt", required=True, type=str)
    parser.add_argument("--target_prompt", required=True, type=str)
    parser.add_argument("--feature_path", type=str, default="feature")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--num_steps", type=int, default=15)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--output_prefix", default="steerflow", type=str)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Control Flow params
    parser.add_argument("--mask_path", type=str, default="")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--upper_quantile", type=float, default=0.8)
    parser.add_argument("--sigmoid_temp", type=float, default=13.0)
    parser.add_argument("--dilation_kernel", type=int, default=5)
    parser.add_argument("--forward_steps", type=int, default=1)

    args = parser.parse_args()
    main(args)
