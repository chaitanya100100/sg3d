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

from config.defaults import get_cfg_defaults
from dataset.ag import AG, cuda_collate_fn
from model.detector import Detector
from model.sttran import STTran
from model.evaluation_recall import BasicSceneGraphEvaluator
from contextlib import redirect_stdout

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
log_file = os.path.join(cfg.TRAIN.EXP_PATH, 'mylog.txt')


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


all_evaluators = {}
for constraint_type in ['with', 'no', 'semi']:
    evaluator =BasicSceneGraphEvaluator(mode=cfg.MODEL.SG_MODE,
                                        AG_object_classes=AG_dataset_train.object_classes,
                                        AG_all_predicates=AG_dataset_train.relationship_classes,
                                        AG_attention_predicates=AG_dataset_train.attention_relationships,
                                        AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                        AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                        iou_threshold=0.5,
                                        semithreshold=0.9,
                                        constraint=constraint_type)
    all_evaluators[constraint_type] = evaluator

if cfg.TRAIN.ONLY_TEST:
    print("Only testing...")
    model.eval()
    with torch.no_grad():
        for b, (batch, index) in tqdm.tqdm(enumerate(dataloader_test)):
            batch = copy.deepcopy(batch)
            gt_anns = copy.deepcopy(dataloader_test.dataset.gt_annotations[index])

            entry = object_detector(batch, gt_anns, is_train=False)
            pred = model(entry)

            for _, evalu in all_evaluators.items():
                evalu.evaluate_scene_graph(gt_anns, pred)

    for k, evalu in all_evaluators.items():
        print(f'-------------------------{k} constraint-------------------------------')
        evalu.print_stats()
    exit(0)
else:
    evaluator = all_evaluators['with']  # Only one type of evaluation during training


# loss function, default Multi-label margin loss
ce_loss = nn.CrossEntropyLoss()
mlm_loss = nn.MultiLabelMarginLoss()


optimizer = AdamW(model.parameters(), lr=cfg.TRAIN.LR)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

for epoch in range(cfg.TRAIN.NUM_EPOCHS):
    model.train()
    start = time.time()
    for b, (batch, index) in enumerate(dataloader_train):
        batch = copy.deepcopy(batch)
        gt_anns = copy.deepcopy(dataloader_train.dataset.gt_annotations[index])

        # prevent gradients to FasterRCNN
        with torch.no_grad():
            entry = object_detector(batch, gt_anns, is_train=True)

        pred = model(entry)

        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]

        attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device).squeeze() # R
        # multi-label margin loss or adaptive loss
        spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=attention_distribution.device)  # R x 6
        contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=attention_distribution.device)  # R x 6
        for i in range(len(pred["spatial_gt"])):
            spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])  # put left-aligned labels. required by MLM loss
            contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

        losses = {}
        if cfg.MODEL.SG_MODE in ['sgcls', 'sgdet']:
            losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

        losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
        losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
        losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)

        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

        log_freq = 500
        if b % log_freq == 0 and b >= log_freq:
            time_per_batch = (time.time() - start) / log_freq
            print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))
            start = time.time()
            torch.cuda.empty_cache()

    torch.save({"state_dict": model.state_dict()}, os.path.join(cfg.TRAIN.EXP_PATH, "model_{}.tar".format(epoch)))
    print("*" * 40)
    with open(log_file, 'a') as f:
        with redirect_stdout(f):
            print("save the checkpoint after {} epochs".format(epoch))

    model.eval()
    with torch.no_grad():
        for b, (batch, index) in tqdm.tqdm(enumerate(dataloader_test)):
            batch = copy.deepcopy(batch)
            gt_anns = copy.deepcopy(dataloader_test.dataset.gt_annotations[index])

            entry = object_detector(batch, gt_anns, is_train=False)
            # entry['h3d_feat'] = copy.deepcopy(data[5].cuda(0))
            pred = model(entry)
            evaluator.evaluate_scene_graph(gt_anns, pred)
        print('-----------', flush=True)
    score = np.mean(evaluator.result_dict[cfg.MODEL.SG_MODE + "_recall"][20])
    with open(log_file, 'a') as f:
        with redirect_stdout(f):
            evaluator.print_stats()
    evaluator.reset_result()
    scheduler.step(score)



