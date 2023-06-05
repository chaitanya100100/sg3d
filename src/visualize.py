import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import sys
import copy
import tqdm
import glob
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from dataset.ag_det2 import ag_categories, ag_rel_classes

from config.defaults import get_cfg_defaults
from dataset.ag import AG, cuda_collate_fn
from model.detector import Detector
from model.sttran import STTran
from utils import visualize_detection, visualize_graph_predictions

import warnings
warnings.filterwarnings('ignore', '.*Byte tensor for key_padding_mask in nn.MultiheadAttention.*', )

import random
random.seed(17)
np.random.seed(17)
torch.random.manual_seed(17)


cfg = get_cfg_defaults()
cfg.merge_from_list(sys.argv[1:])
cfg.freeze()
print(cfg.dump())
if not os.path.exists(cfg.TRAIN.EXP_PATH):
    os.makedirs(cfg.TRAIN.EXP_PATH)


AG_dataset_train = AG(cfg, mode='train')
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(cfg, mode='test')
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")
# freeze the detection backbone
object_detector = Detector(cfg).to(device=gpu_device)
object_detector.eval()

roi_box_pooler = copy.deepcopy(object_detector.model.roi_heads.box_pooler)

model = STTran(
    cfg=cfg,
    roi_box_pooler=roi_box_pooler,
    attention_class_num=len(AG_dataset_train.attention_relationships),
    spatial_class_num=len(AG_dataset_train.spatial_relationships),
    contact_class_num=len(AG_dataset_train.contacting_relationships),
    obj_classes=AG_dataset_train.object_classes,
    ).to(device=gpu_device)


ckpt_path = cfg.MODEL.CKPT_PATH
if ckpt_path == 'last_ckpt':
    ckpt_path = sorted(glob.glob(os.path.join(cfg.TRAIN.EXP_PATH, 'model_*.tar')))[-1]
if ckpt_path != '':
    print(f"Loading ckpt from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=True)


for stype in ['train', 'val']:
    DatasetCatalog.register(f"ag_{stype}", lambda: [])
    MetadataCatalog.get(f"ag_{stype}").set(thing_classes=ag_categories)


out_dir = os.path.join(cfg.TRAIN.EXP_PATH, 'vis')
os.makedirs(out_dir, exist_ok=True)

model.eval()
for itr, (batch, index) in tqdm.tqdm(enumerate(dataloader_test)):
    if itr > 200:
        break
    batch = copy.deepcopy(batch)
    gt_anns = copy.deepcopy(dataloader_test.dataset.gt_annotations[index])

    with torch.no_grad():
        entry = object_detector(batch, gt_anns, is_train=False)
        pred = model(entry)

    for i in range(len(batch)):
        det_img = visualize_detection(cfg.MODEL.SG_MODE, batch, pred, [i])[0]
        pred_graph = visualize_graph_predictions(pred, [i], vis_gt=False)[0]
        gt_graph = visualize_graph_predictions(pred, [i], vis_gt=True)[0]
    
        name = batch[i]['name']
        cv2.imwrite(os.path.join(out_dir, f'{name}_00_gt_graph.jpg'), gt_graph)
        cv2.imwrite(os.path.join(out_dir, f'{name}_01_det.jpg'), det_img)
        cv2.imwrite(os.path.join(out_dir, f'{name}_02_pred_graph.jpg'), pred_graph)
    
