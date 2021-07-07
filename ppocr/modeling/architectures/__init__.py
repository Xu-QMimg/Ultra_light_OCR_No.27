# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#此处报错,
#1增加build_model_kd,无效……


import copy

__all__ = ['build_model']

def build_model(config):
    from .base_model import BaseModel
    
    config = copy.deepcopy(config)#深拷贝，为什么用深拷贝？
    module_class = BaseModel(config)
    return module_class

def build_model_kd(config):
    from .base_model import BaseModel
    
    config_kd = copy.deepcopy(config)#深拷贝，为什么用深拷贝？
    module_class_kd = BaseModel(config_kd)
    return module_class_kd