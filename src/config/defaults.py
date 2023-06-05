from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.DATA_ROOT = '/vision/u/chpatel/data/actiongenome/ActionGenome/dataset/ag/'
_C.DATA.SMALL_SUBSET = False
_C.DATA.FILTER_NONPERSON_BOX_FRAME = True
_C.DATA.FILTER_SMALL_BOX = True
_C.DATA.H3D_GT_BBOX_PATH = '/vision/u/chpatel/data/actiongenome/phalp_out_gt/'
_C.DATA.H3D_PRED_BBOX_PATH = '/vision/u/chpatel/data/actiongenome/phalp_out_det/'
_C.DATA.NUM_WORKERS = 4

_C.MODEL = CN()
_C.MODEL.SUBJECT_FEATURE = 'img'  # {img, h3d, img_h3d}
_C.MODEL.SG_MODE = 'predcls'
_C.MODEL.NUM_ENC_LAYERS = 1
_C.MODEL.NUM_DEC_LAYERS = 3
_C.MODEL.REMOVE_DUPLICATE_PRED_OBJECTS = True

_C.MODEL.DETECTOR_TYPE = 'r50_finetune'
_C.MODEL.CKPT_PATH = ''

_C.TRAIN = CN()
_C.TRAIN.LR = 1.e-5
_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.EXP_PATH = '/vision/u/chpatel/test/test/'
_C.TRAIN.ONLY_TEST = False


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()