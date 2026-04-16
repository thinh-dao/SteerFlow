
<div align="center">

<h1> SteerFlow (Diffusers) </h1>

</div>

> Diffusers-based implementation of SteerFlow. Supports FLUX and Stable Diffusion 3.


<h1> Environment </h1>

```shell
conda create -n steerflow python==3.10
conda activate steerflow
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install diffusers transformers accelerate protobuf sentencepiece
pip install sam3  # for mask generation
```


<h1> Usage </h1>

**FLUX Editing:**

```shell
python demo_edit.py \
    --model_type FLUX \
    --model_path black-forest-labs/FLUX.1-dev \
    --image_path examples/input.jpg \
    --mask_path examples/mask.png \
    --src_prompt "a photo of a cat sitting on a couch" \
    --tar_prompt "a photo of a dog sitting on a couch" \
    --save_folder outputs/
```

**SD3 Editing:**

```shell
python demo_edit.py \
    --model_type SD3 \
    --model_path stabilityai/stable-diffusion-3-medium-diffusers \
    --image_path examples/input.jpg \
    --mask_path examples/mask.png \
    --src_prompt "a photo of a cat sitting on a couch" \
    --tar_prompt "a photo of a dog sitting on a couch" \
    --save_folder outputs/
```

**Generating SAM3 Masks:**

```shell
python generate_masks.py \
    --image_path examples/input.jpg \
    --prompt "cat" \
    --output_path examples/mask.png
```


<h1> Key Parameters </h1>

| Parameter | FLUX Default | SD3 Default | Description |
|-----------|-------------|-------------|-------------|
| `--T_steps` | 15 | 28 | Number of ODE steps |
| `--alpha` | 5.0 | 5.0 | Edit strength decay exponent |
| `--tar_guidance_scale` | 3.5 | 3.5 | Target guidance scale |
