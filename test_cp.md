- use smaller anchor box
- rpn post nms top n = 100

"Adding all other things like spatial mask, glove emb, bbox pose emb didn't improve it. Adding modifications to detector config didn't improve it. Multilinear map is running but doesn't seem to help. Result collection refactored. Experiment with longer training of detector and without subject feature is running."


 - [05/11] Jingwei's code borrows an old but standard repository for image SG generation (https://github.com/jwyang/graph-rcnn.pytorch) and made modifications for his paper. It will be hard to get it working without readme and structured code. One way is to start from image SG repository and borrow Jingwei's addition carefully into this to retrain. Second option is to get working code from some recent papers on video SG generation (given below) as a baseline.
- [05/11] After Jingwei's paper, there have been few papers on video scene graph generation. Most notable are:
  - CVPR 2023 "Unbiased Scene Graph Generation in Videos"
  - ICCV 2021 "Spatial-Temporal Transformer for Dynamic Scene Graph Generation" (They have well structured code: https://github.com/yrcong/STTran/tree/main)
  - WACV 2023 "Exploiting Long-Term Dependencies for Generating Dynamic Scene Graphs" (https://github.com/Shengyu-Feng/DSG-DETR)
  - PAMI 2023 "RelTR: Relation Transformer for Scene Graph Generation" (https://github.com/yrcong/RelTR/tree/main)
  - Cognition Guided Human-Object Relationship Detection
  - ICLR 2023 Video Scene Graph Generation from Single-Frame Weak Supervision https://github.com/zjucsq/PLA
  - DDS: Decoupled Dynamic Scene-Graph Generation Network
  - Meta Spatio-Temporal Debiasing for Video Scene Graph Generation
  - Classification-Then-Grounding: Reformulating Video Scene Graphs As Temporal Bipartite Graphs
  - Dynamic Scene Graph Generation via Anticipatory Pre-training

- [05/10] HORT pretrained backbone and object detection head on visual genome and then action genome, and then kept it fixed.
- [05/10] phalp_out by default code only saves human detections, not all detections.
- [05/08] Got things running on moma-lrg and charades. Confs and job specs for both added. Code to run only phalp or slam commented as we can just use run_opt.py with appropriate arguments. 


 python src/train.py  TRAIN.EXP_PATH /vision/u/chpatel/test/mysg_detr101scratch_sgcls_addall  MODEL.SG_MODE sgcls MODEL.DETECTOR_TYPE r101_scratch MODEL.USE_GLOVE_EMB True MODEL.USE_BBOX_POS_EMB True MODEL.USE_SPATIAL_MASK True

