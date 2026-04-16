
<div align="center">

<h1> SteerFlow (FLUX) </h1>

</div>

> Implementation of SteerFlow based on the official FLUX repository.


<h1> Environment </h1>

```shell
conda create -n steerflow python==3.10
conda activate steerflow

cd YOUR_WORKSPACE/SteerFlow/SteerFlow_FLUX
pip install -e ".[all]"
```


<h1> Usage </h1>

```shell
cd YOUR_WORKSPACE/SteerFlow/SteerFlow_FLUX/src

python edit.py \
    --source_prompt "a photo of a cat sitting on a couch" \
    --target_prompt "a photo of a dog sitting on a couch" \
    --source_img_dir path/to/input.jpg \
    --mask_path path/to/mask.png \
    --output_dir output/ \
    --sampling_strategy steerflow \
    --num_steps 15 \
    --alpha 5.0 \
    --guidance 3.5
```


<h1> Key Parameters </h1>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_steps` | 15 | Number of ODE steps |
| `--alpha` | 5.0 | Edit strength decay exponent |
| `--guidance` | 3.5 | Target guidance scale |
| `--mask_path` | "" | Path to SAM3-generated mask |
| `--sigmoid_temp` | 13.0 | Sigmoid temperature for adaptive mask |
| `--dilation_kernel` | 5 | Morphological kernel size |
| `--forward_steps` | 1 | Number of forward steps for target velocity (0 to disable) |
