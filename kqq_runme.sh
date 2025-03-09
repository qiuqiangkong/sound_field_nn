CUDA_VISIBLE_DEVICES=0 python train.py --config="./kqq_configs/01a.yaml" --no_log
CUDA_VISIBLE_DEVICES=0 python train2.py --config="./kqq_configs/01a.yaml" --no_log

# train.py  # auto-regressively prediction 
# train2.py  # direction prediction the t-th frame

# train3.py  # online data generation