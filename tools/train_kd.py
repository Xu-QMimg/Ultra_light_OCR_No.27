"""
注意
1、加入了教师模型eval模块
Global.pretrained_model_teacher、Architecture_Teacher在教师模型中用到
2、Global.pretrained_model和Global.infer_mode的值在引用教师模型后要改回来！！！！
前者为了载入预训练教师模型，后者修改模式
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import yaml
import paddle
import paddle.distributed as dist

paddle.seed(2)

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
import tools.program as program

dist.get_world_size()


def main(config, device, logger, vdl_writer):
    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()

    global_config = config['Global']

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # build post process后处理
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model构建模型
    #教师网络
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture_Teacher']["Head"]['out_channels'] = char_num
    model_teacher = build_model(config['Architecture_Teacher'])#改为build_model_kd
    
    pretrained_model_path = global_config['pretrained_model']
    global_config['pretrained_model'] = global_config['pretrained_model_teacher']
    init_model(config, model_teacher, logger)#加载模型

    global_config['infer_mode'] = True #模型infer模式

    model_teacher.eval()


    # for rec algorithm学生网络训练

    global_config['infer_mode'] = False
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])#报错，已修改
    if config['Global']['distributed']:
        model = paddle.DataParallel(model)

    # build loss建立损失函数
    loss_class = build_loss(config['Loss'])

    # build optim建立优化器
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())

    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model加载预训练模型
    global_config['pretrained_model']=pretrained_model_path
    pre_best_model_dict = init_model(config, model, logger, optimizer)

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))
    # start train开始训练,loss_class是使用的损失函数
    program.train_kd(config, train_dataloader, valid_dataloader, device, model, model_teacher,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer)


def test_reader(config, device, logger):
    loader = build_dataloader(config, 'Train', device, logger)
    import time
    starttime = time.time()
    count = 0
    try:
        for data in loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader: {}, {}, {}".format(
                    count, len(data[0]), batch_time))
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    main(config, device, logger, vdl_writer)
    # test_reader(config, device, logger)
