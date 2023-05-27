"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn

from model.word_vectors import obj_edge_vectors
from model.transformer import transformer


class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet'):
        super(ObjectClassifier, self).__init__()
        self.mode = mode

    def forward(self, entry):
        if self.mode  == 'predcls':
            entry['pred_labels'] = entry['labels']
            return entry
        raise NotImplementedError


class STTran(nn.Module):

    def __init__(self, mode='sgdet',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None, rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None):

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
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode

        self.object_classifier = ObjectClassifier(mode=self.mode)

        ###################################
        # Relationship head: Conv on RoI features
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
        print("NOT LOADING EMBEDDING VECTORS")
        # embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data', wv_dim=200)
        # self.obj_embed.weight.data = embed_vecs.clone()
        # self.obj_embed2.weight.data = embed_vecs.clone()

        self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter')

        self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(1936, self.contact_class_num)

        self.use_h3d = False
        if self.use_h3d:
            # self.h3d_fc = nn.Linear(2048+99, 512)
            self.h3d_fc = nn.Linear(2048+99+2048, 512)
            print("USING H3D features in STTran")

    def forward(self, entry):

        entry = self.object_classifier(entry)

        # visual part
        if self.use_h3d:
            # subj_rep = self.h3d_fc(entry['h3d_feat'])
            # subj_rep = subj_rep[entry['im_idx']]

            inp = [entry['h3d_feat'][entry['im_idx']], entry['features'][entry['pair_idx'][:, 0]]]
            subj_rep = self.h3d_fc(torch.cat(inp, 1))
        else:
            subj_rep = entry['features'][entry['pair_idx'][:, 0]]
            subj_rep = self.subj_fc(subj_rep)
            
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        # vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])  # R x 256 x 7 x 7
        vr = self.union_func1(entry['union_feat']) #+self.conv(entry['spatial_masks'])  # R x 256 x 7 x 7
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

