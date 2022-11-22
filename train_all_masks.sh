set -e

CUDA_VISIBLE_DEVICES=7 python train.py --c config/exp_cv/fixmatch_cifar100_400_0_mask0.1.yaml

CUDA_VISIBLE_DEVICES=7 python train.py --c config/exp_cv/fixmatch_cifar100_400_0_mask0.2.yaml

CUDA_VISIBLE_DEVICES=3 python train.py --c config/exp_cv/fixmatch_cifar100_400_0_mask0.3.yaml

# CUDA_VISIBLE_DEVICES=5 python train.py --c config/exp_cv/fixmatch_cifar100_400_0_mask0.4.yaml

# CUDA_VISIBLE_DEVICES=7 python train.py --c config/exp_cv/fixmatch_cifar100_400_0_mask0.5.yaml

# CUDA_VISIBLE_DEVICES=1 python train.py --c config/exp_cv/fixmatch_cifar100_400_0_mask0.6.yaml

# CUDA_VISIBLE_DEVICES=2 python train.py --c config/exp_cv/fixmatch_cifar100_400_0_mask0.7.yaml