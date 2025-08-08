# partly cited from refers
__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

import json
from types import SimpleNamespace

json_cfg = """{
        "LOSS": {
            "WEIGHT_POWER": 1.1,
            "EXTRA_WEIGHT": [1.0, 1.0],
            "SCHEDULER": "re_weight",
            "DRW_EPOCH": 50,
            "CLS_EPOCH_MIN": 20,
            "CLS_EPOCH_MAX": 60,
            "MWNL": {
                "GAMMA": 2.0,
                "BETA": 0.1,
                "TYPE": "fix",
                "SIGMOID": "normal"
            }
        }
    }
    """
# type in [zero, fix, decrease]
# simoid in [normal, enlarge]
# 将 JSON 字符串解析为 Python 对象
config_dict = json.loads(json_cfg)

# 将字典转换为 SimpleNamespace
def convert_to_namespace(d):
    return SimpleNamespace(**{k: convert_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})

cfg = convert_to_namespace(config_dict)


if __name__ == '__main__':
    import torch
    from loss import MWNLoss
    # ��ʼ�� MWNLoss
    para_dict = {
        "num_class_list": [100, 50],  # ÿ����������������TODO��
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # �豸��CPU �� GPU��
        "cfg": cfg
        }
    mwn_loss = MWNLoss(para_dict=para_dict)

    # ����������������������Ŀ������
    x = torch.randn(16, 2).to(para_dict["device"])  # ������������״Ϊ (batch_size, num_classes)
    target = torch.randint(0, 2, (16,)).to(para_dict["device"])  # Ŀ����������״Ϊ (batch_size,)

    # ������ʧ
    loss = mwn_loss(x, target)
    print(loss)