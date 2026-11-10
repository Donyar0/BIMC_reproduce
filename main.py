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
    cfg.TRAINER.BiMC.VISION_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_I = -1.0
    cfg.TRAINER.BiMC.TAU = -1
    cfg.TRAINER.BiMC.TEXT_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_T = -1.0
    cfg.TRAINER.BiMC.GAMMA_BASE = -1.0
    cfg.TRAINER.BiMC.GAMMA_INC = -1.0
    cfg.TRAINER.BiMC.USING_ENSEMBLE = False

    # Optional training-free local patch classifier (CDPS).
    cfg.TRAINER.CDPS = CN()
    cfg.TRAINER.CDPS.ENABLED = False
    cfg.TRAINER.CDPS.GLOBAL_WEIGHT = 0.5
    cfg.TRAINER.CDPS.TOP_PERCENT = 0.4
    cfg.TRAINER.CDPS.NUM_REGIONS = 2
    cfg.TRAINER.CDPS.TOPK = 20
    cfg.TRAINER.CDPS.NUM_CONFUSE = 5
    cfg.TRAINER.CDPS.OMEGA = 0.1
    cfg.TRAINER.CDPS.MAX_SUPPORT = 8
    cfg.TRAINER.CDPS.MAX_QUERY_PATCHES = 64
    cfg.TRAINER.CDPS.QUERY_TOPK = 20
    cfg.TRAINER.CDPS.TEMPERATURE = 1.0
    cfg.TRAINER.CDPS.CACHE_DIR = "output/cdps_patch_memory"




    

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
    parser.add_argument('--use_swanlab', action='store_true', help="Enable SwanLab experiment logging")
    parser.add_argument('--swanlab_project', type=str, default='BIMC_reproduce', help="SwanLab project name")
    parser.add_argument('--swanlab_experiment', type=str, default='', help="SwanLab experiment name")
    parser.add_argument('--swanlab_mode', type=str, default='online', choices=['online', 'offline', 'local', 'disabled'],
                        help="SwanLab running mode")

    args = parser.parse_args()

    data_cfg = args.data_cfg
    train_cfg = args.train_cfg

    cfg = setup_cfg(data_cfg, train_cfg)
    init_swanlab(cfg, args)

    # Set the random seed and GPU ID
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    # Import and run the trainer
    from engine.engine import Runner
    engine = Runner(cfg)
    try:
        engine.run()
    finally:
        finish()


if __name__ == '__main__':
    main()
