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
    cfg.FLOW.EPOCHS = 100
    cfg.FLOW.BATCH_SIZE = 64
    cfg.FLOW.LR = 0.0002
    cfg.FLOW.WEIGHT_DECAY = 0.0001
    cfg.FLOW.BLOCKS = 24
    cfg.FLOW.HIDDEN = 0
    cfg.FLOW.TARGET = 'prototype'
    cfg.FLOW.SOURCE = 'sample'
    cfg.FLOW.CE_WEIGHT = 0.0
    cfg.FLOW.TEMPERATURE = 0.07
    cfg.FLOW.STEPS = 8
    cfg.FLOW.ALPHAS = [1.0]
    cfg.FLOW.LOGIT_GAMMAS = [0.0]  # subtract gamma from base-class cosine scores
    cfg.FLOW.EVAL_ONLY = False
    cfg.FLOW.CHECKPOINT = ''
    cfg.FLOW.TRUST_CHECKPOINT = False
    cfg.FLOW.OUTPUT_DIR = 'output/single_flow'
    cfg.FLOW.META = CN()
    cfg.FLOW.META.ENABLED = False
    cfg.FLOW.META.EPOCHS = 20
    cfg.FLOW.META.EPISODES_PER_EPOCH = 30
    cfg.FLOW.META.PSEUDO_NEW = 10
    cfg.FLOW.META.SHOT = 5
    cfg.FLOW.META.QUERY = 5
    cfg.FLOW.META.BASE_BATCH = 64
    cfg.FLOW.META.OLD_QUERY_BATCH = 64
    cfg.FLOW.META.INNER_STEPS = 1
    cfg.FLOW.META.INNER_LR = 0.0001
    cfg.FLOW.META.INNER_CE_WEIGHT = 0.1
    cfg.FLOW.META.LR = 0.0001
    cfg.FLOW.META.FM_WEIGHT = 1.0
    cfg.FLOW.META.OLD_CE_WEIGHT = 0.5
    cfg.FLOW.META.QUERY_CE_WEIGHT = 1.0
    cfg.FLOW.INC = CN()
    cfg.FLOW.INC.ENABLED = False
    cfg.FLOW.INC.EPOCHS = 10
    cfg.FLOW.INC.BATCH_SIZE = 25
    cfg.FLOW.INC.LR = 1e-5
    cfg.FLOW.INC.WEIGHT_DECAY = 0.0
    cfg.FLOW.INC.CE_WEIGHT = 0.1
    cfg.FLOW.INC.TEMPERATURE = 0.07




    

def setup_cfg(dataset_cfg_file, method_cfg_file):
    cfg = CN()
    extend_cfg(cfg)

    # 1. From the dataset config file
    cfg.merge_from_file(dataset_cfg_file)

    # 2. From the method config file
    cfg.merge_from_file(method_cfg_file)

    cfg.freeze()
    return cfg


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the pipeline")

    parser.add_argument('--data_cfg', type=str, help="Path to the data configuration file")
    parser.add_argument('--train_cfg', type=str, help="Path to the training configuration file")
    parser.add_argument('--flow_checkpoint', type=str, default=None,
                        help="Load an existing visual-to-text Flow and skip training")
    parser.add_argument('--flow_alphas', type=float, nargs='+', default=None,
                        help="Visual-to-text residual strengths, e.g. 0 0.1 0.2 0.5 1")
    parser.add_argument('--flow_logit_gammas', type=float, nargs='+', default=None,
                        help='Base-class cosine subtraction strengths, e.g. 0 0.005 0.01 0.02 0.05 0.1')
    parser.add_argument('--flow_trusted_checkpoint', action='store_true',
                        help="Allow pickle loading of your own trusted Flow checkpoint")
    parser.add_argument('--use_swanlab', action='store_true', help="Enable SwanLab experiment logging")
    parser.add_argument('--swanlab_project', type=str, default='BIMC_reproduce', help="SwanLab project name")
    parser.add_argument('--swanlab_experiment', type=str, default='', help="SwanLab experiment name")
    parser.add_argument('--swanlab_mode', type=str, default='online', choices=['online', 'offline', 'local', 'disabled'],
                        help="SwanLab running mode")

    args = parser.parse_args()

    data_cfg = args.data_cfg
    train_cfg = args.train_cfg

    cfg = setup_cfg(data_cfg, train_cfg)
    if (args.flow_checkpoint is not None or args.flow_alphas is not None
            or args.flow_logit_gammas is not None or args.flow_trusted_checkpoint):
        if cfg.METHOD.lower() != 'visual_to_text':
            parser.error('--flow_* overrides require METHOD=visual_to_text')
        cfg.defrost()
        if args.flow_checkpoint is not None:
            cfg.FLOW.CHECKPOINT = args.flow_checkpoint
        if args.flow_alphas is not None:
            cfg.FLOW.ALPHAS = args.flow_alphas
        if args.flow_logit_gammas is not None:
            cfg.FLOW.LOGIT_GAMMAS = args.flow_logit_gammas
        if args.flow_trusted_checkpoint:
            cfg.FLOW.TRUST_CHECKPOINT = True
        cfg.freeze()
    init_swanlab(cfg, args)

    # Set the random seed and GPU ID
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    # Import and run the trainer
    if cfg.METHOD.lower() == 'single_flow':
        from engine.flow_engine import FlowRunner as Runner
    elif cfg.METHOD.lower() == 'visual_to_text':
        from engine.visual_to_text_engine import VisualToTextRunner as Runner
    elif cfg.METHOD.lower() in ('', 'bimc'):
        from engine.engine import Runner
    else:
        raise ValueError(f'Unknown METHOD: {cfg.METHOD}')
    engine = Runner(cfg)
    try:
        engine.run()
    finally:
        finish()


if __name__ == '__main__':
    main()



