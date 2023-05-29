import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
# from scipy.misc import imread
from imageio import imread
import joblib
import numpy as np
import pickle
import os
import copy
import joblib
from collections import defaultdict

from .ag_det2 import ag_categories, ag_rel_classes, process_one_frame, get_config_ag


from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import detection_utils
from detectron2.data import transforms as det_transforms
class MyDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        detection_utils.check_image_size(dataset_dict, image)

        aug_input = det_transforms.AugInput(image)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class SingletonAGData(object):
    """A fancy singleton class to avoid loading data repeatedly."""
    instance = None
    def __new__(cls, data_root, prefix, suffix):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
            person_bbox = joblib.load(os.path.join(data_root, f'annotations/{prefix}person_bbox.pkl'))
            object_bbox = joblib.load(os.path.join(data_root, f'annotations/{prefix}object_bbox_and_relationship{suffix}.pkl'))
            resolution = joblib.load(os.path.join(data_root, 'annotations/resolution.pkl'))

            # bad ann correction
            num_bad = 0
            for i, (k, v) in enumerate(object_bbox.items()):
                vname = k.split('/')[0]
                res = np.array(resolution[vname])
                for o in v:
                    if o['bbox'] is None: continue
                    bbox = np.array(o['bbox'])
                    mini = bbox[:2]
                    maxi = bbox[:2]+bbox[2:]
                    outrange = ~((mini >= 0) & (maxi <= res))
                    num_bad += int(any(outrange))
                    if any(outrange):
                        mini = res - bbox[2:]
                        wd = res - bbox[:2]
                        if any(mini < 0): bbox[2:] = wd
                        else:             bbox[:2] = mini
                        o['bbox'] = tuple(bbox)
            print("{} bad object bbox corrected".format(num_bad))
            cls.instance.person_bbox = person_bbox
            cls.instance.object_bbox = object_bbox
            cls.instance.resolution = resolution

        return cls.instance


class AG(Dataset):

    def __init__(self, cfg, mode):
        self.cfg = cfg

        self.sg_mode = cfg.MODEL.SG_MODE
        self.use_h3d = (cfg.MODEL.SUBJECT_FEATURE in ['h3d', 'img_h3d'])
        self.data_root = cfg.DATA.DATA_ROOT
        self.filter_small_box = cfg.DATA.FILTER_SMALL_BOX and (self.sg_mode != 'predcls')

        self.h3d_path = cfg.DATA.H3D_PRED_BBOX_PATH if self.sg_mode == 'sgdet' else cfg.DATA.H3D_GT_BBOX_PATH
        self.frames_path = os.path.join(self.data_root, 'frames/')

        self.object_classes = ag_categories
        self.relationship_classes = ag_rel_classes
        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]


        print('-------loading annotations---------slowly-----------')
        prefix = 'small_' if cfg.DATA.SMALL_SUBSET else ''
        suffix = '_filtersmall' if self.filter_small_box else ''
        agdata = SingletonAGData(self.data_root, prefix, suffix)
        person_bbox = copy.deepcopy(agdata.person_bbox)
        object_bbox = copy.deepcopy(agdata.object_bbox)
        resolution = copy.deepcopy(agdata.resolution)
        print('--------------------finish!-------------------------')
        self.person_bbox = copy.deepcopy(person_bbox)
        self.object_bbox = copy.deepcopy(object_bbox)
        self.resolution = resolution

        # collect valid frames
        video_dict = defaultdict(list)  # <vid>: ["<vid>/<fid1>", "<vid>/<fid2>", ...]
        for i in person_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = any(j['visible'] for j in object_bbox[i])
                if frame_valid:
                    video_dict[i.split('/')[0]].append(i)
        
        """
        self.gt_annotations = list of vid_ann
        vid_ann = list of frame_ann
        frame_ann = [{'person_bbox': <bbox>}, obj1_ann, obj2_ann]
        objn_ann = {
            'class': cid, 
            'bbox': xyxy, 
            'attention/spatial/contacting_relationship': longtensor,
        }
        """
        self.video_list = []  # [ [v1f1, v1f2,...], [v2f1, v2f2, v2f3, ...], ...]
        self.gt_annotations = []
        
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0
        self.valid_nums = 0
        self.h3d_frames = 0
        self.non_h3d_frames = 0

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                if cfg.DATA.FILTER_NONPERSON_BOX_FRAME:
                    if person_bbox[j]['bbox'].shape[0] == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self.valid_nums += 1


                gt_annotation_frame = [{'person_bbox': person_bbox[j]['bbox'], 'class': 0}]
                # each frames's objects and human
                for k in object_bbox[j]:
                    if k['visible']:
                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
                        k['class'] = self.object_classes.index(k['class'])
                        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
                        k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
                        k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
                        k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            if len(video) > 2:
                self.video_list.append(video)
                self.gt_annotations.append(gt_annotation_video)

                # code snippet to find the info about h3d predictions. It takes longer time. Only use it to get statistics.
                # h3d_info = joblib.load(os.path.join(self.h3d_path, video[0].split('/')[0], 'results/demo_.pkl'))
                # for v in video:
                #     has_emb = int(len(h3d_info[os.path.join(self.data_root, 'frames', v)]['embedding']) > 0)
                #     self.h3d_frames += has_emb
                #     self.non_h3d_frames += 1-has_emb

            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        print('x'*60)
        print("DISCARDING VIDEOS WITH 2 FRAMES ???...")  # see the if else right above
        # print("Frames with h3d: {}, without h3d {}".format(self.h3d_frames, self.non_h3d_frames))
        if cfg.DATA.FILTER_NONPERSON_BOX_FRAME:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
            # print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(non_heatmap_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format("<not available>"))
        print('x' * 60)

        # Data mapper for feeding images to detectron2 detector.
        dcfg = get_config_ag()
        self.data_mapper = MyDatasetMapper(dcfg, is_train=False)

    def __getitem__(self, index):

        frame_names = self.video_list[index]
        
        if self.use_h3d:
            h3d_info = joblib.load(os.path.join(self.h3d_path, frame_names[0].split('/')[0] + '.pkl'))

        batch = []
        for idx, name in enumerate(frame_names):

            anns, _, _ = process_one_frame(self.person_bbox[name], self.object_bbox[name])
            width, height = self.resolution[name.split('/')[0]]
            
            sample = {
                'file_name': os.path.join(self.frames_path, name),
                'annotations': anns,
                'width': width, 'height': height,
            }
            # Data mapper resizes image according to test min/max size config.
            # It also creates Instances from annotations.
            sample = self.data_mapper(sample)

            if self.use_h3d:
                human_info = h3d_info[name]  # a list of human info
                if len(human_info):
                    assert human_info[0]['embedding'].shape[0] == 4096+2048+2048+99  # [ 4096 appearance, 2048+2048 pose, 95 location ]
                    fr_h3d_feat =  human_info[0]['embedding'][-2048-99:]  # take pose and location embedding of first person
                else:
                    # no human found. take zero embedding. This shouldn't happen for gt bbox but may happen for pred bbox
                    # if detector can't find the human. This seems to be happening less than 5% of the times.
                    fr_h3d_feat = torch.zeros(2048+99)
                sample['h3d_feat'] = fr_h3d_feat
            batch.append(sample)

        return batch, index

    def __len__(self):
        return len(self.video_list)


def cuda_collate_fn(batch):
    return batch[0]
