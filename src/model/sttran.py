"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn

from model.word_vectors import obj_edge_vectors
from model.transformer import transformer
from detectron2.structures import Boxes
from model.draw_rectangles.draw_rectangles import draw_union_boxes


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)"""
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] + 0.5 * wh, wh))
    return torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)

class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, sg_mode, num_classes, roi_box_pooler, remove_duplicate_pred_objects, obj_glove_emb, use_bbox_pos_emb, use_spatial_mask):
        super(ObjectClassifier, self).__init__()
        self.sg_mode = sg_mode
        self.num_classes = num_classes
        self.remove_duplicate_pred_objects = remove_duplicate_pred_objects
        self.use_spatial_mask = use_spatial_mask

        if self.sg_mode == 'predcls':
            return

        self.obj_embed = nn.Embedding(num_classes, 200)
        if obj_glove_emb is not None:
            self.obj_embed.weight.data = obj_glove_emb.clone()
            print("Using GLOVE vectors in object classifier")
        else:
            print("Not loading GLOVE vectors in object classifier")

        self.use_bbox_pos_emb = use_bbox_pos_emb
        if self.use_bbox_pos_emb:
            # This probably doesn't help it much
            self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                        nn.Linear(4, 128),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.1))

        self.decoder_lin = nn.Sequential(nn.Linear(1024 + 200 + int(self.use_bbox_pos_emb) * 128, 1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, num_classes))
        self.roi_box_pooler = roi_box_pooler

    def forward_predcls(self, entry):
        entry['pred_labels'] = entry['gt_labels']
        entry['pred_scores'] = entry['gt_scores']
        return entry

    def forward_sgcls(self, entry):
        # Use detector distribution to get obj label embedding and then use features again with a head
        # to get new object classification logits. This head will be trained. Since detector already
        # gives good classification, this part seems redundant.
        obj_embed = entry['distribution'] @ self.obj_embed.weight  # B x F
        if self.use_bbox_pos_emb:
            pos_e = self.pos_embed(center_size(entry['boxes'][:, 1:]))  # B x F: doesn't help much
            obj_features = torch.cat((entry['features'], obj_embed, pos_e), 1)
        else:
            obj_features = torch.cat((entry['features'], obj_embed), 1)
        entry['distribution'] = self.decoder_lin(obj_features)

        if self.training:
            # Use GT labels during training to train transformer. We will use predicted labels during testing.
            entry['pred_labels'] = entry['gt_labels']
            entry['pred_scores'] = entry['gt_scores']
        else:
            box_idx = entry['boxes'][:,0].long() # frame_idx of each bbox
            b = int(box_idx[-1] + 1)  # total number of frames

            # Get predicted distribution
            entry['distribution'] = torch.softmax(entry['distribution'], dim=-1)
            # Discard person class and predict labels and score. Even human bbox will have an object prediction.
            # We will then explicitly find human bbox in each frame and overwrite.
            entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=-1)
            entry['pred_labels'] = entry['pred_labels'] + 1

            # Use distribution to get the person bbox for each frame
            human_idx = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
            global_idx = torch.arange(0, entry['boxes'].shape[0])
            local_human_idx = [0]*b
            for i in range(b):
                lhi = torch.argmax(entry['distribution'][box_idx == i, 0]) # the local bbox index with highest human score in this frame
                human_idx[i] = global_idx[box_idx == i][lhi]
                local_human_idx[i] = lhi.item()
            
            # But now assign human label to predicted human boxes (overwrite object labels assigned previously to those bboxes)
            entry['pred_labels'][human_idx.squeeze()] = 0
            entry['pred_scores'][human_idx.squeeze()] = entry['distribution'][human_idx.squeeze(), 0]

            if self.remove_duplicate_pred_objects:
                # This finds the duplicate object class for each frame and replaces it with second best predicted class. Hacky!
                for i in range(b):
                    present = entry['boxes'][:, 0] == i
                    duplicate_class = torch.mode(entry['pred_labels'][present])[0]
                    if torch.sum(entry['pred_labels'][present] ==duplicate_class) > 0:
                        duplicate_position = entry['pred_labels'][present] == duplicate_class
                        assert duplicate_class.item() >= 0 and duplicate_class.item() < self.num_classes
                        ppp = torch.argsort(entry['distribution'][present][duplicate_position][:,duplicate_class])[:-1]
                        for j in ppp:
                            changed_idx = global_idx[present][duplicate_position][j]
                            entry['distribution'][changed_idx, duplicate_class] = 0
                            entry['pred_labels'][changed_idx] = torch.argmax(entry['distribution'][changed_idx])
                            entry['pred_scores'][changed_idx] = torch.max(entry['distribution'][changed_idx])

            # Create relations with predicted humans. entry already has relation specific tensors but they are made using
            # gt labels. During testing, we should use predicted labels to do everything. Hence overwrite those tensors.
            im_idx = []  # which frame are the relations belong to
            pair = [] # R x 2
            pair_single = [] # N x [r x 2]
            for j, i in enumerate(human_idx):
                for m in global_idx[box_idx==j][entry['pred_labels'][box_idx==j] != 0]: # this long term contains the other objects in the frame
                    im_idx.append(j)
                    pair.append([int(i), int(m)])
                
                pair_single.append([])
                for m in torch.nonzero(entry['pred_labels'][box_idx==j] != 0): 
                    pair_single[j].append([local_human_idx[j], int(m)])

            for i in range(len(pair_single)): pair_single[i] = torch.tensor(pair_single[i]).long()

            pair = torch.tensor(pair).to(obj_features.device)
            im_idx = torch.tensor(im_idx, dtype=torch.long).to(obj_features.device)
            entry['pair_idx'] = pair
            entry['im_idx'] = im_idx
            entry['pair_single'] = pair_single

            # Now use predicted potential relations to get union boxes and features.
            union_boxes = []
            pair_boxes = [] # R x 8
            for i, (ps, boxs) in enumerate(zip(pair_single, entry['scaled_pred_boxes'])):
                if ps.max().item() >= boxs.shape[0]:
                    print("in object classifier", ps, boxs)
                    exit(0)
                sb, ob = boxs[ps[:,0]], boxs[ps[:,1]]
                pair_boxes.append(torch.cat([sb, ob], -1))
                ub = torch.cat([torch.min(sb, ob)[:,:2], torch.max(sb, ob)[:,2:]], 1)
                union_boxes.append(ub)
            with torch.no_grad():
                union_features = self.roi_box_pooler(entry['prepool_features'], [Boxes(ub) for ub in union_boxes])
                # union_features = self.model.roi_heads.box_head(union_features)  # B x F
                spatial_masks = None
                if self.use_spatial_mask:
                    spatial_masks = torch.tensor(draw_union_boxes(torch.cat(pair_boxes).cpu().detach().numpy(), 27) - 0.5).to(union_features.device)
            entry['union_feat'] = union_features
            entry['spatial_masks'] = spatial_masks
        return entry

    def forward(self, entry):
        if self.sg_mode  == 'predcls':
            return self.forward_predcls(entry)
        elif self.sg_mode == 'sgcls':
            return self.forward_sgcls(entry)
        else:
            raise NotImplementedError


class STTran(nn.Module):

    def __init__(self, cfg, roi_box_pooler=None,
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None, rel_classes=None):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(STTran, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.sg_mode = cfg.MODEL.SG_MODE
        assert self.sg_mode in ('sgdet', 'sgcls', 'predcls')

        obj_glove_emb = None
        if cfg.MODEL.USE_GLOVE_EMB:
            obj_glove_emb = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='/vision/u/chpatel/data/glove/', wv_dim=200)

        self.use_spatial_mask = cfg.MODEL.USE_SPATIAL_MASK
        self.object_classifier = ObjectClassifier(
            cfg.MODEL.SG_MODE, num_classes=len(obj_classes), roi_box_pooler=roi_box_pooler,
            remove_duplicate_pred_objects=cfg.MODEL.REMOVE_DUPLICATE_PRED_OBJECTS,
            obj_glove_emb=obj_glove_emb, use_bbox_pos_emb=cfg.MODEL.USE_BBOX_POS_EMB, use_spatial_mask=self.use_spatial_mask)

        ###################################
        # Relationship head: Conv on RoI union box features
        self.union_func1 = nn.Conv2d(256, 256, 1, 1)
        # self.union_func1 = nn.Conv2d(1024, 256, 1, 1)

        # Conv to process spatial mask of union bbox
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        # self.subj_fc = nn.Linear(2048, 512)
        # self.obj_fc = nn.Linear(2048, 512)
        self.vr_fc = nn.Linear(256*7*7, 512)
        self.subj_fc = nn.Linear(1024, 512)
        self.obj_fc = nn.Linear(1024, 512)

        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        if obj_glove_emb is not None:
            self.obj_embed.weight.data = obj_glove_emb.clone()
            self.obj_embed2.weight.data = obj_glove_emb.clone()
            print("Using GLOVE vectors in STTran")
        else:
            print("Not loading GLOVE vectors in STTran")

        self.glocal_transformer = transformer(
            enc_layer_num=cfg.MODEL.NUM_ENC_LAYERS, dec_layer_num=cfg.MODEL.NUM_DEC_LAYERS, 
            embed_dim=1936, nhead=8, dim_feedforward=2048, dropout=0.1, mode='latter')

        self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(1936, self.contact_class_num)

        # What features to use for human subject
        self.subject_feature_type = cfg.MODEL.SUBJECT_FEATURE
        if self.subject_feature_type == 'h3d':
            print("Using H3D features")
            self.h3d_fc = nn.Linear(2048+99, 512)
        elif self.subject_feature_type == 'img_h3d':
            print("Using Image and H3D features")
            self.h3d_fc = nn.Linear(1024+2048+99, 512)
        elif self.subject_feature_type == 'img_h3d_mlm':
            print("Using Multilinear map with Image and H3D features")
            self.mlm = RandomizedMultiLinearMap(2048+99, 1024, 2048)
            self.h3d_fc = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        elif self.subject_feature_type == 'none':
            print("Using No Subject Features")
            pass
        else:
            assert self.subject_feature_type == 'img'


    def forward(self, entry):

        entry = self.object_classifier(entry)

        # visual part
        if self.subject_feature_type == 'h3d':
            subj_rep = self.h3d_fc(entry['h3d_feat'])  # N x F1 to N x F2
            subj_rep = subj_rep[entry['im_idx']] # R x F2 using rel2im_idx

        elif self.subject_feature_type == 'img_h3d':
            inp = [entry['h3d_feat'][entry['im_idx']], 
                   entry['features'][entry['pair_idx'][:, 0]]]
            subj_rep = self.h3d_fc(torch.cat(inp, 1))
        
        elif self.subject_feature_type == 'img_h3d_mlm':
            subj_rep = self.mlm(entry['h3d_feat'][entry['im_idx']], entry['features'][entry['pair_idx'][:, 0]])
            subj_rep = self.h3d_fc(subj_rep)      
        elif self.subject_feature_type == 'none':
            subj_rep = torch.zeros(entry['im_idx'].shape[0], 512, 
                                   device=entry['features'].device, dtype=entry['features'].dtype)
        else:
            subj_rep = entry['features'][entry['pair_idx'][:, 0]]
            subj_rep = self.subj_fc(subj_rep)
            
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        # vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])  # R x 256 x 7 x 7
        vr = self.union_func1(entry['union_feat']) #+self.conv(entry['spatial_masks'])  # R x 256 x 7 x 7
        if self.use_spatial_mask:
            vr += self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1) # R x 1536 where 512*3=1536

        # semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)  # R x 400

        rel_features = torch.cat((x_visual, x_semantic), dim=1) # R x 1936
        # Spatial-Temporal Transformer
        global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'])

        entry["attention_distribution"] = self.a_rel_compress(global_output)
        entry["spatial_distribution"] = self.s_rel_compress(global_output)
        entry["contacting_distribution"] = self.c_rel_compress(global_output)

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])
        # attention_distribution is still score, not distribution
        # spatial/conntacting_distribution is per class probability of having that rel
        return entry


class RandomizedMultiLinearMap(torch.nn.Module):

    def __init__(self, feat_dim, num_classes, map_dim = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.scaler = np.sqrt(float(map_dim))
        Rf = torch.randn(feat_dim, map_dim)
        Rg = torch.randn(num_classes, map_dim)
        self.register_buffer('Rf', Rf, persistent=True)
        self.register_buffer('Rg', Rg, persistent=True)
        self.output_dim = map_dim

    def forward(self, f, g):
        f = f @ self.Rf
        g = g @ self.Rg
        output = (f * g) / self.scaler
        return output