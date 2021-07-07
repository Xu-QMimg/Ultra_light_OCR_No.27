#Add from .rec_distill_loss import KLLoss
#New loss name "KLLoss"

import copy


def build_loss(config):
    # det loss
    from .det_db_loss import DBLoss
    from .det_east_loss import EASTLoss
    from .det_sast_loss import SASTLoss

    # rec loss此处添加蒸馏损失
    from .rec_ctc_loss import CTCLoss
    from .rec_att_loss import AttentionLoss
    from .rec_srn_loss import SRNLoss
    from .rec_distill_loss import KLLoss

    # cls loss
    from .cls_loss import ClsLoss

    # e2e loss
    from .e2e_pg_loss import PGLoss
    support_dict = [
        'DBLoss', 'EASTLoss', 'SASTLoss', 'CTCLoss', 'ClsLoss', 'AttentionLoss',
        'SRNLoss', 'PGLoss','KLLoss']

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('loss only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)#执行一个字符串表达式，并返回表达式的值。eval(expression, globals=None, locals=None)
    return module_class
