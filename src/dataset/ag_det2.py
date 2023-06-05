import detectron2
from detectron2.engine import DefaultTrainer

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
import os
import joblib
import tqdm
import imagesize


ag_categories = ['person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 
          'chair', 'closet/cabinet', 'clothes', 'cup/glass/bottle', 'dish', 'door', 'doorknob', 
          'doorway', 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 
          'paper/notebook', 'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich', 
          'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel', 'vacuum', 'window']
ag_rel_classes = ['looking_at', 'not_looking_at', 'unsure', 
                  'above', 'beneath', 'in_front_of', 'behind',  'on_the_side_of', 'in', 
                  'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back',
                  'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship', 'sitting_on', 
                  'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on']


def save_small_subset_ag():
    import random
    import joblib
    import os

    root_path = '/vision/u/chpatel/data/actiongenome/ActionGenome/dataset/ag/'
    person_bbox = joblib.load(os.path.join(root_path, 'annotations/person_bbox.pkl'))
    obj_bbox = joblib.load(os.path.join(root_path, 'annotations/object_bbox_and_relationship.pkl'))

    vid_ids = list(set([k.split('/')[0] for k in person_bbox.keys()]))
    random.seed(17)
    random.shuffle(vid_ids)
    vid_ids = vid_ids[:10]

    small_person_bbox = {}
    small_obj_bbox = {}
    for k in person_bbox.keys():
        if k.split('/')[0] in vid_ids:
            small_person_bbox[k] = person_bbox[k]
            small_obj_bbox[k] = obj_bbox[k]

    joblib.dump(small_obj_bbox, os.path.join(root_path, 'annotations/small_object_bbox_and_relationship.pkl'))
    joblib.dump(small_person_bbox, os.path.join(root_path, 'annotations/small_person_bbox.pkl'))

    small_obj_bbox = joblib.load(os.path.join(root_path, 'annotations/small_object_bbox_and_relationship.pkl'))
    obj_bbox_fs = joblib.load(os.path.join(root_path, 'annotations/object_bbox_and_relationship_filtersmall.pkl'))
    small_obj_bbox_fs = {}
    for k in small_obj_bbox.keys():
        small_obj_bbox_fs[k] = obj_bbox_fs[k]
    joblib.dump(small_obj_bbox_fs, os.path.join(root_path, 'annotations/small_object_bbox_and_relationship_filtersmall.pkl'))


def process_one_frame(pbbox, obbox, stats=None):
    is_train = True
    objs = []

    if stats is None:
        stats = {
            'num_invi_obj' : 0,
            'num_visi_obj' : 0,
            'num_invi_person' : 0,
            'num_visi_person' : 0,
            'num_obj_frames' : 0,
            'num_missed_frames' : 0,
        }

    # iterate over each object
    for od in obbox:
        if not od['visible']:
            stats['num_invi_obj'] += 1
            continue

        is_train = (od['metadata']['set'] == 'train')
        obj = {
            "bbox": od['bbox'],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": ag_categories.index(od['class']),
        }
        objs.append(obj)
    
    # track num frames with no objs
    stats['num_visi_obj'] += len(objs)
    stats['num_obj_frames'] += int(len(objs) != 0)
    
    # add person as object
    if len(pbbox['bbox']) == 0:
        stats['num_invi_person'] += 1
    else:
        obj = {
            "bbox": tuple(pbbox['bbox'][0]),
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": ag_categories.index('person'),
        }
        objs.insert(0, obj)
        stats['num_visi_person'] += 1
    
    return objs, is_train, stats


def get_ag_dicts(root_path, small_subset=False):
    
    prefix = 'small_' if small_subset else ''
    if small_subset:
        print("USING SMALLER SUBSET OF ACTION GENOME FOR DEBUGGING")

    person_bbox = joblib.load(os.path.join(root_path, f'annotations/{prefix}person_bbox.pkl'))
    print("AG person bbox loaded.")
    obj_bbox = joblib.load(os.path.join(root_path, f'annotations/{prefix}object_bbox_and_relationship.pkl'))
    print("AG obj bbox loaded.")

    resolution = joblib.load(os.path.join(root_path, 'annotations/resolution.pkl'))

    train_dicts = []
    val_dicts = []
    
    stats = None
    
    # iterate over annotated frames
    for vf in tqdm.tqdm(person_bbox.keys()):
        
        objs, is_train, stats = process_one_frame(person_bbox[vf], obj_bbox[vf], stats)
        
        # skip if there is no object at all
        if len(objs) == 0:
            stats['num_missed_frames'] += 1
            continue
            
        # create image and annotations for a frame
        file_name = os.path.join(root_path, 'frames', vf)
        width, height = resolution[vf.split('/')[0]]
        # width, height = imagesize.get(file_name)
        record = {
            'image_id': vf,
            'file_name': file_name,
            'annotations': objs,
            'height': height,
            'width': width,
        }
        
        if is_train: train_dicts.append(record)
        else: val_dicts.append(record)

    print("Train size", len(train_dicts))
    print("val size", len(val_dicts))
    print(stats)
    return train_dicts, val_dicts


def register_action_genome():
    # Register AG dataset
    train_dicts, val_dicts = get_ag_dicts('/vision/u/chpatel/data/actiongenome/ActionGenome/dataset/ag/')
    for dcts, stype in [[train_dicts, 'train'], [val_dicts, 'val']]:
        DatasetCatalog.register(f"ag_{stype}", lambda: dcts)
        MetadataCatalog.get(f"ag_{stype}").set(thing_classes=ag_categories)
        MetadataCatalog.get(f"ag_{stype}").set(evaluator_type='coco')


def get_config_ag_detector(dtype):
    if dtype == 'r50_finetune':
        cfg = get_config_ag()
        cfg.OUTPUT_DIR = "/vision/u/chpatel/test/faster_rcnn_ag_2_contd"
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.MODEL.WEIGHTS = "/vision/u/chpatel/test/faster_rcnn_ag_2_contd/model_final.pth"
    elif dtype == 'r50_scratch':
        cfg = get_config_ag()
        cfg.OUTPUT_DIR = "/vision/u/chpatel/test/faster_rcnn_ag_scratch"
        cfg.MODEL.WEIGHTS = "/vision/u/chpatel/test/faster_rcnn_ag_scratch/model_final.pth"
    elif dtype == 'r101_scratch':
        cfg = get_config_ag()
        cfg.MODEL.RESNETS.DEPTH = 101
        cfg.SOLVER.IMS_PER_BATCH = 24
        cfg.SOLVER.BASE_LR = 0.0003
        # cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        # cfg.SOLVER.MAX_ITER = 1000
        # cfg.SOLVER.CHECKPOINT_PERIOD = 500
        cfg.OUTPUT_DIR = "/vision/u/chpatel/test/faster_rcnn_ag_scratch_r101"
        cfg.MODEL.WEIGHTS = "/vision/u/chpatel/test/faster_rcnn_ag_scratch_r101/model_final.pth"
    else:
        raise NotImplementedError
    return cfg


def get_config_ag():

    # Prep config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("ag_train",)
    cfg.DATASETS.TEST = ("ag_val",)

    # Keep min and max size consistent in train and test
    cfg.INPUT.MIN_SIZE_TRAIN = (480,)
    cfg.INPUT.MIN_SIZE_TEST = 480

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 36
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]. STTran had somehow [4,8,16,32]
    # cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 100
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300

    cfg.SOLVER.IMS_PER_BATCH = 32
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (30000, 60000, 90000)
    cfg.SOLVER.MAX_ITER = 120000
    cfg.SOLVER.CHECKPOINT_PERIOD = 15000

    cfg.OUTPUT_DIR = "/vision/u/chpatel/test/test"

    return cfg

"""
## TRAIN
Frames with h3d: 170693, without h3d 6442 (pred bbox)
There are 7584 videos and 177330 valid frames
144 videos are invalid (no person), remove them
49 videos are invalid (only one frame), remove them
21643 frames have no human bbox in GT, remove them!

## TEST
Frames with h3d: 54958, without h3d 1922 (pred bbox)
There are 1750 videos and 56923 valid frames
41 videos are invalid (no person), remove them
19 videos are invalid (only one frame), remove them
8636 frames have no human bbox in GT, remove them!
"""