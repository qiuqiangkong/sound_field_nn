# Sound filed prediction with neural network

## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/sound_field_nn
cd sfnn

# Install Python environment
conda create --name sfnn python=3.10

# Activate environment
conda activate sfnn

# Install Python packages dependencies
bash env.sh
```

## Step 1: FDTD simulator

Run the following script will demonstrate 2D sound field simulator.

```python
python sound_field_nn/data/fdtd2d.py
```

## Step 2: NN prediction

```python
CUDA_VISIBLE_DEVICES=0 python train3.py --config="./kqq_configs/03a.yaml" --no_log
```