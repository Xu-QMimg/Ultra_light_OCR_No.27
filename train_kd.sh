python3 -m paddle.distributed.launch --gpus '0,1,2,3,4' ./tools/train_kd.py -c ./configs/distillation.yml
