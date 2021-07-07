# Ultra_light_OCR_No.27
轻量级文字识别技术创新大赛B榜27名

## 简介
这里是[Paddle轻量级文字识别技术创新大赛](https://aistudio.baidu.com/aistudio/competition/detail/75)第27名的代码链接。
- 推断模型**大小9.39M，A榜准确度74.3%，B榜准确度73.9%**
- 模型整体沿用MobileNetV3_small，训练策略等均未改变，**只使用知识蒸馏的方法提高准确度**
- 模型及训练日志：[百度网盘链接](https://pan.baidu.com/s/1y5cGG6CtZ4OI0BHrt2iosw)，提取码：f8pl 

## 目录

## 算法介绍

## 环境部署
- **环境部署和[PPOCR](https://github.com/PaddlePaddle/PaddleOCR)完全一致**
- python =3.7
- PaddlePaddle-gpu = 2.0.2
```
python3.7 -m pip install -r requirements.txt
```

## 测试
下载模型[百度网盘链接](https://pan.baidu.com/s/1y5cGG6CtZ4OI0BHrt2iosw)，提取码：f8pl 
```
cd Ultra_light_OCR_No.27
python tools/infer_rec.py -c "/output/distllation/config.yml" -o Global.infer_img="/data/OCRimages/TestBImages/" Global.pretrained_model="/output/distllation/best_accuracy"
```

## 训练
### step1 数据准备：请自行下载比赛数据集或参考[训练文件](https://github.com/simplify23/PaddleOCR/blob/release/2.1/doc/doc_ch/recognition.md )
如果需要自定义，请一并修改配置文件
- 训练集路径：
```
|-dataset
  |-train
    |- labeltrain.txt    #标签
    |- Train_000000.jpg
    |- Train_000001.jpg
    |- ...
```

- 测试集路径：
```
|-dataset
  |-test
    |- Test_000000.jpg
    |- ...
```

### step2: 启动教师模型训练
```
cd Ultra_light_OCR_No.27
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1' ./tools/train.py -c ./configs/distillation/rec_chinese_common_train_v2.0.yml
```

### step3: 启动学生模型训练
```
cd Ultra_light_OCR_No.27
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1' ./tools/train_kd.py -c ./configs/distillation.yml
```

### step4: 导出模型
```
python3 tools/export_model.py -c "/output/distllation/config.yml" -o Global.pretrained_model="/output/distllation/best_accuracy" Global.save_inference_dir=./inference/distllation
```
### step5：预测
```
python tools/infer_rec.py -c "/output/distllation/config.yml" -o Global.infer_img="/dataset/test/" Global.pretrained_model="/output/distllation/best_accuracy"
```

## License

[MIT](LICENSE) © Richard Littauer
