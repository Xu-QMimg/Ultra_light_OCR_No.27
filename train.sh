# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch --gpus '0,1,2' ./tools/train.py -c ./configs/distillation/rec_chinese_common_train_v2.0.yml