# Evaluation

type | AUC-Judd/Borji/shuffled  | CC | NSS
---- | ---- | --- | ---
This repo. | 0.8308/0.7752/0.7100 | 0.6961 | 1.6015


## How to run:

``` python
# train
CUDA_VISIBLE_DEVICES=1 python train.py \
  --batch_size=1 \
  --mode='train' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --g_bce \
  --n_dis=1 \
  --input_dir=/home/tellhow-iot/tem/webpagesaliency/train \
  --output_dir=/data/tem/webpagesaliency/output \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=1180 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=572 \
  --TTUR \
  --l1_weight=0.05 \
  --gan_weight=1.0 \
  --multiple_A \
  --net_type='ResNet' \
  --upsampe_method='depth_to_space'


# test
CUDA_VISIBLE_DEVICES=1 python train.py \
  --batch_size=1 \
  --mode='test' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --g_bce \
  --n_dis=1 \
  --input_dir=/home/tellhow-iot/tem/webpagesaliency/val \
  --output_dir=/data/tem/webpagesaliency/output/tem/47200 \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=1180 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=512 \
  --TTUR \
  --l1_weight=0.05 \
  --gan_weight=1.0 \
  --checkpoint_dir=/data/tem/webpagesaliency/output/ \
  --checkpoint=/data/tem/webpagesaliency/output/model-47200 \
  --multiple_A \
  --net_type='ResNet' \
  --upsampe_method=depth_to_space
