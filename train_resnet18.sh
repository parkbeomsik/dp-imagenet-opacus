
export OMP_NUM_THREADS=2

# accuracy=41 on 50th epoch
# /home/beomsik/miniconda3/envs/dp-imagenet-opacus/bin/python imagenet/imagenet_train.py \
#   --data_dir imagenet/imagenet-data \
#   --train_device_batch_size=256 \
#   --eval_device_batch_size=1024 \
#   --eval_every_n_steps=1024 \
#   --model=resnet18 \
#   --num_train_epochs=70 \
#   --dp_clip_norm=1.0 \
#   --dp_sigma=0.058014 \
#   --grad_acc_steps=64 \
#   --base_learning_rate=0.03 \
#   --lr_warmup_epochs=1 \
#   --num_layers_to_freeze=7 \
#   --finetune_path="pretrained_models/places365_resnet18_20220314.npz" \
#   --multiprocessing_distributed \
#   --print_freq 64

# 41.630

taskset -c 40-79 /home/beomsik/miniconda3/envs/dp-imagenet-opacus/bin/python imagenet/imagenet_train.py \
  --data_dir imagenet/imagenet-data \
  --train_device_batch_size=128 \
  --eval_device_batch_size=1024 \
  --eval_every_n_steps=1024 \
  --model=resnet18 \
  --num_train_epochs=70 \
  --dp_clip_norm=1.0 \
  --dp_delta=8e-7 \
  --grad_acc_steps=128 \
  --base_learning_rate=0.03 \
  --lr_warmup_epochs=1 \
  --num_layers_to_freeze=7 \
  --finetune_path="pretrained_models/places365_resnet18_20220314.npz" \
  --multiprocessing_distributed \
  --print_freq 10000 \
  --rnd_seed 4 \
  --port 12316

