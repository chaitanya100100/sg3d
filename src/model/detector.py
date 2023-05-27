import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import cv2
import os

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.boxes import Boxes


from dataset.ag_det2 import get_config_ag
from utils import print_dict


def make_relations(human_idx, gt_anns):
    rel2im_idx = []  # which frame are the relations belong to
    pair = []  # R x 2 from list of [human_bbox_idx, obj_bbox_idx]
    a_rel = []  # R x 1 long tensor [ [type_of_attn], [type_of_attn], [type_of_attn] ]
    s_rel = []  # R lengthed list of lists [ list_of_spa_rel, list_of_spa_rel, list_of_spa_rel]
    c_rel = []
    pair_single = []  # a list of rx2 relations - one for each frame

    bidx = 0
    for i, fr_ann in enumerate(gt_anns):
        pair_single.append([])
        for j, ob_a in enumerate(fr_ann):
            if 'person_bbox' in ob_a.keys():
                bidx += 1
            else:
                rel2im_idx.append(i)
                pair.append([human_idx[i].item(), bidx])
                pair_single[i].append([0, j])
                a_rel.append(ob_a['attention_relationship'].tolist())
                s_rel.append(ob_a['spatial_relationship'].tolist())
                c_rel.append(ob_a['contacting_relationship'].tolist())
                bidx += 1

    rel2im_idx = torch.tensor(rel2im_idx, dtype=torch.long)
    pair = torch.tensor(pair).long()
    for i in range(len(pair_single)): pair_single[i] = torch.tensor(pair_single[i]).long()

    return rel2im_idx, pair, pair_single, a_rel, s_rel, c_rel


class Detector(nn.Module):

    '''first part: object detection (image/video)'''

    def __init__(self, cfg):
        super(Detector, self).__init__()

        self.sg_mode = cfg.MODEL.SG_MODE
        assert self.sg_mode == 'predcls'

        # Load pretrained detector
        dcfg = get_config_ag()
        dcfg.freeze()
        model = build_model(dcfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.DETECTOR_WEIGHTS)
        model.cuda().eval()
        self.model = model

        self.num_classes = dcfg.MODEL.ROI_HEADS.NUM_CLASSES
        # self.decoder_lin = nn.Sequential(nn.Linear(1024, 1024),
        #                                  nn.BatchNorm1d(1024),
        #                                  nn.ReLU(),
        #                                  nn.Linear(1024, self.num_classes))

    def forward(self, batch, gt_anns, is_train):

        device = self.model.device
        batch_size = 10
        with torch.no_grad():
            big_batch = batch
            big_pred_boxes = []
            big_box_features = []
            big_pred_scores = []
            big_pred_classes = []
            big_scaled_image_sizes = []
            big_prepool_features = []
            for i in range(0, len(big_batch), batch_size):
                batch = big_batch[i:i+batch_size]
                scaled_images = self.model.preprocess_image(batch)
                scaled_image_sizes = scaled_images.image_sizes
                features = self.model.backbone(scaled_images.tensor)
                prepool_features = [features[f] for f in self.model.roi_heads.box_in_features]

                """
                Bboxes
                        | training | Inference
                        |----------|----------
                predcls |   gt     |    gt
                sgcls   |   gt     |    gt
                sgdet   |
                """

                if self.sg_mode != 'predcls':
                    proposals, _ = self.model.proposal_generator(scaled_images, features, None)
                    scaled_instances, _ = self.model.roi_heads(scaled_images, features, proposals, None)

                    # Remove empty boxes. In default model.forward, this step is done in model._postprocess
                    scaled_instances = [x[x.pred_boxes.nonempty()] for x in scaled_instances]
                    pred_boxes = [x.pred_boxes for x in scaled_instances]
                    pred_scores = torch.cat([x.scores for x in scaled_instances])
                    pred_classes = torch.cat([x.pred_classes for x in scaled_instances])
                else:
                    scaled_instances = [b['instances'].to(device) for b in batch]
                    pred_boxes = [x.gt_boxes for x in scaled_instances]
                    pred_classes = torch.cat([x.gt_classes for x in scaled_instances])
                    pred_scores = torch.ones_like(pred_classes).float()

                box_features = self.model.roi_heads.box_pooler(prepool_features, pred_boxes)
                box_features = self.model.roi_heads.box_head(box_features)  # B x F
                pred_boxes = [b.tensor for b in pred_boxes]  # N x [b x 4]

                big_pred_boxes.extend(pred_boxes)
                big_scaled_image_sizes.extend(scaled_image_sizes)
                big_box_features.append(box_features)
                big_pred_scores.append(pred_scores)
                big_pred_classes.append(pred_classes)
                big_prepool_features.append(prepool_features)
            
        pred_boxes = big_pred_boxes
        scaled_image_sizes = big_scaled_image_sizes
        batch = big_batch
        box_features = torch.cat(big_box_features, dim=0)
        pred_scores = torch.cat(big_pred_scores, dim=0)
        pred_classes = torch.cat(big_pred_classes, dim=0)
        prepool_features = [torch.cat(pf, dim=0) for pf in zip(*big_prepool_features)]

        distribution, _ = self.model.roi_heads.box_predictor(box_features)
        distribution = F.softmax(distribution, dim=-1)

        """
        Object labels and distribution
                | training | Inference
                |----------|----------
        predcls |   gt     |    gt
        sgcls   |   gt     |    pred
        sgdet   |
        """

        # Prepare for relationship
        num_boxes_per_im = torch.tensor([len(x) for x in pred_boxes], dtype=torch.long)  # N
        box2img_idx = torch.repeat_interleave(torch.arange(len(batch), dtype=torch.long), num_boxes_per_im) # B
        human_idx = torch.nonzero(pred_classes == 0).view(-1)  # should be N

        rel2im_idx, pair, pair_single, a_rel, s_rel, c_rel = make_relations(human_idx, gt_anns)
        rel2im_idx = rel2im_idx.to(device)
        pair = pair.to(device)

        # Get Union boxes and features
        union_boxes = []
        for i, (ps, boxs) in enumerate(zip(pair_single, pred_boxes)):
            if ps.max().item() >= boxs.shape[0]:
                print(batch[i], ps, boxs)
                exit(0)
            sb, ob = boxs[ps[:,0]], boxs[ps[:,1]]
            ub = torch.cat([torch.min(sb, ob)[:,:2], torch.max(sb, ob)[:,2:]], 1)
            union_boxes.append(ub)

        with torch.no_grad():
            union_features = self.model.roi_heads.box_pooler(prepool_features, [Boxes(ub) for ub in union_boxes])
            # union_features = self.model.roi_heads.box_head(union_features)  # B x F

        # rescale bboxes to original image size
        og_pred_boxes = []
        for scsz, pb, bt in zip(scaled_image_sizes, pred_boxes, batch):
            scale_x, scale_y = bt['width'] / scsz[1],  bt['height'] / scsz[0]
            pb = pb.clone()
            pb[:, 0::2] *= scale_x
            pb[:, 1::2] *= scale_y
            og_pred_boxes.append(pb)
        ret_boxes = torch.cat([box2img_idx.to(device)[:, None], torch.cat(og_pred_boxes)], 1)

        out = {
            'boxes': ret_boxes,  # B x 5
            'labels': pred_classes,
            'scores': pred_scores,
            'im_idx': rel2im_idx,
            'pair_idx': pair, # R x 2
            'features': box_features,  # bbox_num x 1024
            'union_feat': union_features,
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel
        }
        # print(out)
        return out



def get_my_output_bkp(mod, bat):
    with torch.no_grad():
        scaled_images = mod.preprocess_image(bat)
        features = mod.backbone(scaled_images.tensor)
        proposals, _ = mod.proposal_generator(scaled_images, features, None)
        
        prepool_features = [features[f] for f in mod.roi_heads.box_in_features]
        #box_features = mod.roi_heads.box_pooler(prepool_features, [x.proposal_boxes for x in proposals])
        #box_features = mod.roi_heads.box_head(box_features)
        #predictions = mod.roi_heads.box_predictor(box_features)
        #scaled_instances, nms_indices = mod.roi_heads.box_predictor.inference(predictions, proposals)
        
        scaled_instances, _ = mod.roi_heads(scaled_images, features, proposals, None)
        #print([len(x) for x in scaled_instances])
        #non_empty_boxes = [x.pred_boxes.nonempty() for x in scaled_instances]
        #print(torch.cat(non_empty_boxes).logical_not().sum())
        scaled_instances = [x[x.pred_boxes.nonempty()] for x in scaled_instances]
        
        det_box_features = mod.roi_heads.box_pooler(prepool_features, [x.pred_boxes for x in scaled_instances])
        det_box_features = mod.roi_heads.box_head(det_box_features)
        det_box_scores, det_box_deltas = mod.roi_heads.box_predictor(det_box_features)
        num_boxes_per_im = [len(x) for x in scaled_instances]
        print(det_box_scores.shape, num_boxes_per_im)
        det_box_scores = det_box_scores[:,:-1].split(num_boxes_per_im, dim=0)
        det_box_deltas = det_box_deltas.split(num_boxes_per_im)
        det_box_features = det_box_features.split(num_boxes_per_im)
        
        # Careful. This overwrites scaled_instances
        instances = mod._postprocess(scaled_instances, bat, scaled_images.image_sizes)
        
        for ist, ds, dd, df in zip(instances, det_box_scores, det_box_deltas, det_box_features):
            ist['detections'] = {
                'det_features': df,
                'det_scores': ds,
                'det_deltas': dd,
            }
        
    return instances