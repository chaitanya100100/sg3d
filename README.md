## VSGG with 3D humans

- This repository contains code for my CS231N project. I have made this repository public for evaluation.
- The detector is based on Detectron2. This repo provides code for detector training as well as using it for VSGG.
- VSGG pipeline is based on STTran. It also has evaluation code to directly compare with baselines.
- I have added an option to use 3D human features as subject features. 3D human features are computed and stored offline for easy of development.
- There is also visualization code.

## Visualization on Stanford Network
- You can directly visit this link: http://macondo2.stanford.edu:8666/cgi-bin/file-explorer/?dir=%2Fvision%2Fu%2Fchpatel%2Ftest%2Fmysg_sgcls%2Fvis&patterns_show=*&patterns_highlight=&w=600&h=600&n=3&autoplay=1&showmedia=1&mr=90-100
  - Change row numbers to visualize more examples. Make sure to note use more than a range of 10 to not overload the visualization server.
- 3D human pose estimation can be visualized here: http://macondo2.stanford.edu:8666/cgi-bin/file-explorer/?dir=%2Fvision%2Fu%2Fchpatel%2Fdata%2Factiongenome%2Fphalp_out_gt_test_render&patterns_show=*&patterns_highlight=&w=600&h=600&n=3&autoplay=1&showmedia=1&mr=50-60
  - As before, don't visualize more than 10 rows at a time.

## Visualization outside Stanford Network
- Some examples of VSGG and 3D pose features can be visualized here: https://drive.google.com/drive/folders/1LOiyEFm0vWN0wsAraOynoXZW0ctiTRtK?usp=sharing
