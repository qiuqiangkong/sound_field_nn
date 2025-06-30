CUDA_VISIBLE_DEVICES=0 python train.py --config="./kqq_configs/04a.yaml" --no_log

CUDA_VISIBLE_DEVICES=0 python inference.py \
	--config="./kqq_configs/04a.yaml" \
	--ckpt_path="./checkpoints/train/04a/step=2000.pth"