from yacs.config import CfgNode as CN
from utils.util import set_gpu, set_seed
from utils.experiment_logger import finish, init_swanlab
import argparse

def print_args(cfg):
    print("************")
    print("** Config **")
    print("************")
    print(cfg)
    print("************")


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """

    # Device setting
    cfg.DEVICE = CN()
    cfg.DEVICE.DEVICE_NAME = ''
    cfg.DEVICE.GPU_ID = ''

    cfg.METHOD = ''
    cfg.SEED = -1

    # For dataset config
    cfg.DATASET = CN()
    cfg.DATASET.NAME = ''
    cfg.DATASET.ROOT = ''
    cfg.DATASET.GPT_PATH = ''
    cfg.DATASET.NUM_CLASSES   = -1
    cfg.DATASET.NUM_INIT_CLS  = -1
    cfg.DATASET.NUM_INC_CLS   = -1
    cfg.DATASET.NUM_BASE_SHOT = -1
    cfg.DATASET.NUM_INC_SHOT  = -1
    cfg.DATASET.BETA = -1.0
    cfg.DATASET.ENSEMBLE_ALPHA = -1.0
    
    # For data
    cfg.DATALOADER = CN()
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE = -1
    cfg.DATALOADER.TRAIN.BATCH_SIZE_INC = -1
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = -1
    cfg.DATALOADER.NUM_WORKERS = -1

    # For model
    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ''

    # For methods
    cfg.TRAINER = CN()
    cfg.TRAINER.BiMC = CN()
    cfg.TRAINER.BiMC.PREC = ''
    cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR = ''
    cfg.TRAINER.BiMC.VISION_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_I = -1.0
    cfg.TRAINER.BiMC.TAU = -1
    cfg.TRAINER.BiMC.TEXT_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_T = -1.0
    cfg.TRAINER.BiMC.GAMMA_BASE = -1.0
    cfg.TRAINER.BiMC.GAMMA_INC = -1.0
    cfg.TRAINER.BiMC.USING_ENSEMBLE = False

    cfg.FLOW = CN()
    cfg.FLOW.STEPS_PER_EPOCH = 100
    cfg.FLOW.VISUAL_SCALE = 1.0
    cfg.FLOW.MAX_RADIUS = 0.5
    cfg.FLOW.COV_SHRINK = 0.1
    cfg.FLOW.COV_TRANSFER_TEMPERATURE = 16.0
    cfg.FLOW.REPLAY_FRACTION = 0.5
    cfg.FLOW.CYCLE_SCALE = 0.01
    cfg.FLOW.CYCLE_GATE = False
    cfg.FLOW.COSINE_TEMPERATURE = 1.0
    cfg.FLOW.KNN_TEMPERATURE = 25.0
    cfg.FLOW.COV_DIVISOR = 512.0
    cfg.FLOW.DISTILL_WEIGHT = 0.0
    cfg.FLOW.REPLAY_STRATEGY = 'base_incremental_half'
    cfg.FLOW.CONTRAST_WEIGHT = 0.0
    cfg.FLOW.CONTRAST_TEMPERATURE = 20.0
    cfg.FLOW.BANK_MODE = 'flow'
    cfg.FLOW.GAMMA_BASE = -1.0
    cfg.FLOW.GAMMA_INC = -1.0
    cfg.FLOW.BIMC_BLEND_LAMBDA = -1.0
    cfg.FLOW.FUNNEL_MODE = False
    cfg.FLOW.SCORE_MODE = 'probability'
    cfg.FLOW.TTA = False
    cfg.FLOW.TTA_PROTOTYPES = False
    cfg.FLOW.DIAGONAL_WHITEN = False
    cfg.FLOW.VIS_WHITEN_POWER = 0.5
    cfg.FLOW.TEXT_WHITEN_MODE = 'visual'
    cfg.FLOW.USE_GEOMETRY_PRIORS = False
    cfg.FLOW.NOISE_SCALE = 1.0
    cfg.FLOW.MAX_ANGLE = 0.5
    cfg.FLOW.MIX_PROBABILITY = 0.2
    cfg.FLOW.EPOCHS = 100
    cfg.FLOW.BATCH_SIZE = 64
    cfg.FLOW.LR = 0.0002
    cfg.FLOW.WEIGHT_DECAY = 0.0001
    cfg.FLOW.BLOCKS = 24
    cfg.FLOW.HIDDEN = 0
    cfg.FLOW.TARGET = 'prototype'
    cfg.FLOW.STEPS = 8
    cfg.FLOW.SOLVER = 'euler'
    cfg.FLOW.OUTPUT_DIR = 'output/tangent_flow'
    cfg.FLOW.ADAPT_STEPS = 100
    cfg.FLOW.ADAPT_LR = 0.0002
    cfg.FLOW.ALPHA = 0.15
    cfg.FLOW.BETA_BASE = 0.1
    cfg.FLOW.BETA_INC = 0.5
    cfg.FLOW.GAMMA = 0.5
    cfg.FLOW.LOGIT_SCALE = 1.0
    cfg.FLOW.PREDICTION = 'velocity'
    cfg.FLOW.TIME_EPS = 0.05
    cfg.FLOW.WARMUP_RATIO = 0.05
    cfg.FLOW.MIN_LR_RATIO = 0.01

    cfg.LORA = CN()
    cfg.LORA.ENABLED = False
    cfg.LORA.RANK = 2
    cfg.LORA.ALPHA = 2.0
    cfg.LORA.LAST_BLOCKS = 4
    cfg.LORA.EPOCHS = 5
    cfg.LORA.LR = 0.0002
    cfg.LORA.WEIGHT_DECAY = 0.0001
    cfg.LORA.ANCHOR_WEIGHT = 0.1
    cfg.LORA.SEMANTIC_REG_WEIGHT = 0.0

    cfg.RIDGE = CN()
    cfg.RIDGE.FEATURE_DIM = 8192
    cfg.RIDGE.LAMBDA = 1.0
    cfg.RIDGE.BATCH_SIZE = 256
    cfg.RIDGE.BLEND_WEIGHT = 0.5
    cfg.RIDGE.TARGET_MODE = 'prototype'

    cfg.GEOMETRY = CN()
    cfg.GEOMETRY.FLOW_TYPE = 'mlp'
    cfg.GEOMETRY.KERNEL_BANDWIDTH = 0.4
    cfg.GEOMETRY.KAPPA = 5.0

    cfg.FEATURE = CN()
    cfg.FEATURE.MULTILAYER = False
    cfg.FEATURE.LAST_LAYERS = 4
    cfg.FEATURE.PATCH_POOLING = False
    cfg.FEATURE.PATCH_TEMPERATURE = 10.0
    cfg.FEATURE.PATCH_WEIGHT = 0.25
    cfg.FEATURE.DESCRIPTION_CHUNK = 256




    

def setup_cfg(dataset_cfg_file, method_cfg_file, opts=None):
    cfg = CN()
    extend_cfg(cfg)

    # 1. From the dataset config file
    cfg.merge_from_file(dataset_cfg_file)

    # 2. From the method config file
    cfg.merge_from_file(method_cfg_file)

    if opts:
        cfg.merge_from_list(opts)

    cfg.freeze()
    return cfg


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the pipeline")

    parser.add_argument('--data_cfg', type=str, help="Path to the data configuration file")
    parser.add_argument('--train_cfg', type=str, help="Path to the training configuration file")
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                        help="YACS overrides as KEY VALUE pairs; place last")
    parser.add_argument('--use_swanlab', action='store_true', help="Enable SwanLab experiment logging")
    parser.add_argument('--swanlab_project', type=str, default='BIMC_reproduce', help="SwanLab project name")
    parser.add_argument('--swanlab_experiment', type=str, default='', help="SwanLab experiment name")
    parser.add_argument('--swanlab_mode', type=str, default='online', choices=['online', 'offline', 'local', 'disabled'],
                        help="SwanLab running mode")

    args = parser.parse_args()

    data_cfg = args.data_cfg
    train_cfg = args.train_cfg

    cfg = setup_cfg(data_cfg, train_cfg, args.opts)
    init_swanlab(cfg, args)

    # Set the random seed and GPU ID
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    # Import and run the trainer
    if cfg.METHOD.lower() == 'frozen_ridge_flow':
        from engine.frozen_ridge_flow_engine import FrozenRidgeFlowRunner as Runner
    elif cfg.METHOD.lower() == 'balanced_kernel_flow':
        from engine.balanced_kernel_flow_engine import BalancedKernelFlowRunner as Runner
    elif cfg.METHOD.lower() == 'tangent_flow':
        from engine.tangent_flow_engine import TangentFlowRunner as Runner
    else:
        raise ValueError('Unknown method: ' + cfg.METHOD)
    engine = Runner(cfg)
    try:
        engine.run()
    finally:
        finish()


if __name__ == '__main__':
    main()

