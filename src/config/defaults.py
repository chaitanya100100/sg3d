from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.DATA_ROOT = '/vision/u/chpatel/data/actiongenome/ActionGenome/dataset/ag/'
_C.DATA.SMALL_SUBSET = False
_C.DATA.FILTER_NONPERSON_BOX_FRAME = True
_C.DATA.FILTER_SMALL_BOX = False
_C.DATA.H3D_PATH = '/vision/u/chpatel/data/actiongenome/phalp_out/'
_C.DATA.NUM_WORKERS = 4

_C.MODEL = CN()
_C.MODEL.USE_H3D = False
_C.MODEL.SG_MODE = 'predcls'
_C.MODEL.NUM_ENC_LAYERS = 1
_C.MODEL.NUM_DEC_LAYERS = 3

_C.MODEL.DETECTOR_WEIGHTS = '/vision/u/chpatel/test/faster_rcnn_ag_2_contd/model_final.pth'

_C.TRAIN = CN()
_C.TRAIN.LR = 1.e-5
_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.EXP_PATH = '/vision/u/chpatel/test/test/'


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()