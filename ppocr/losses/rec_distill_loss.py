#蒸馏损失


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn


class KLLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(KLLoss, self).__init__()
        self.loss_func1 = nn.CTCLoss(blank=0, reduction='none')
        #self.loss_func2 = nn.functional.kl_div(reduction='mean', name=None)#加入KL损失
        self.loss_func2 =nn.KLDivLoss(reduction='mean')#加入KL损失
        self.log_softmax=nn.LogSoftmax(axis=0, name=None)#log_softmax,对第一个维度的做处理
        self.softmax=nn.Softmax(axis=0, name=None)#softmax

    def __call__(self, predicts_teacher,predicts, batch,T,exp,epoch):#五个输入
        alpha_0=0.8
        alpha=alpha_0
        #衰减系数的设置
        #0.995:200次衰减到0.36，500次衰减到0.08
        #0.994:200此衰减到0.3，500次衰减到0.04
        #0.993:200此衰减到0.24，500次衰减到0.02
        #0.99：200次0.13
        for i in range(epoch):
            alpha=alpha*0.995        


        predicts = predicts.transpose((1, 0, 2))#数据重排(80,batch_size,6625)
        predicts_teacher = predicts_teacher.transpose((1, 0, 2))#数据重排(80,batch_size,6625)
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor([N] * B, dtype='int64')
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        loss1 = self.loss_func1(predicts, labels, preds_lengths, label_lengths)
        loss2 = T * T * self.loss_func2(self.log_softmax(predicts/T),self.softmax(predicts_teacher/T))#加入KL损失
        loss2=loss2*exp #10|100|1000
        #loss=loss1+loss2
        loss=loss1*(1-alpha)+loss2*alpha
        loss = loss.mean()  # sum
        #return {'loss': loss}
        return {'loss': loss,'Hard_loss':loss1,'KL_loss':loss2}
        #return {'loss': loss,'Hard_loss':loss1,'KL_loss':loss2,'alpha':alpha}