import torch

class Config:
    # yaml configs
    EPOCHS= 30
    WARMUP_EPOCH= 5
    BASE_LR= 2.5e-4
    WEIGHT_DECAY= 1e-8
    MIN_LR= 1e-5
    LAYER_LR_DECAY= 0.8
    EVAL_BN_WHEN_TRAINING: True

    # own configs
    DEVICE = "cuda:5"
    BATCH_SIZE = 256 
    




    @classmethod
    def as_dict(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if k.isupper() and not k.startswith("_")
        }