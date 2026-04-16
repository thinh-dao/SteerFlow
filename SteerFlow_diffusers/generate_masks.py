import argparse
import os

import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def generate_mask(image_path: str, prompt: str, output_path: str):
    """Generate a segmentation mask using SAM3 with a text prompt."""
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output["masks"]
    if hasattr(masks, "cpu"):
        masks_tensor = masks.cpu()
    else:
        masks_tensor = torch.from_numpy(masks)

    if masks_tensor.ndim == 4:
        masks_tensor = masks_tensor.squeeze(1)

    merged_mask = torch.any(masks_tensor > 0, dim=0)
    mask_uint8 = (merged_mask.numpy().astype("uint8") * 255)
    mask_image = Image.fromarray(mask_uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mask_image.save(output_path)
    print(f"Mask saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate SAM3 segmentation masks")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for segmentation")
    parser.add_argument("--output_path", type=str, default="mask.png")
    args = parser.parse_args()

    generate_mask(args.image_path, args.prompt, args.output_path)
