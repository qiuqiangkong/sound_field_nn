# Sound filed prediction with neural network

## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/sound_field_nn
cd sound_field_nn

# Install Python environment
conda create --name sound_field_nn python=3.10

# Activate environment
conda activate sound_field_nn

# Install Python packages dependencies
bash env.sh
```

## 1. Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/cnn.yaml" --no_log
```

## 2. Inference
```python
CUDA_VISIBLE_DEVICES=0 python inference.py \
	--config="./configs/cnn.yaml" \
	--ckpt_path="./checkpoints/train/cnn/step=10000.pth"

```

## Results
