# Augmenting Efficient Surgical Instrument Segmentation in Video with Point Tracking and Segment Anything (SIS-PT-SAM)

## News
This work won the <strong>Outstanding Paper Award</strong> at MICCAI 2024 AE-CAI Workshop!

The work was accepted by MICCAI 2024 AE-CAI Workshop!

PyTorch implementation of the SIS-PT-SAM. Inference speed achieves 25+/80+ FPS on single RTX 4060/4090 GPU. Use point prompts for full fine-tuning MobileSAM.

[[arXiv](https://arxiv.org/abs/2403.08003)]

<!-- ## Description

An in-depth paragraph about your project and overview of use. -->

<img src="assets/cholecseg8k_demo_8.gif" height="180"> <img src="assets/stir_demo.gif" height="180"> <img src="assets/ucl_demo_13.gif" height="180">
## Getting Started
### Dependencies

* Python 3.11
* torch 2.1.2
* torchvision 0.16.2

### Environment Setup
Create a Conda environment:
```
conda create --name sis-pt-sam python=3.11
```

Activate the Conda environment and run the following command to install environment
```
pip install -r requirements.txt
```

### Datasets
Links to the publicly available dataset used in this work:
* [EndoVis 2015](https://endovissub-instrument.grand-challenge.org/)
* [EndoVis 2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)
* [EndoVis 2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/home/)
* [ROBUST-MIS 2019](https://www.synapse.org/Synapse:syn18779624/wiki/592660)
* [AutoLaparo](https://autolaparo.github.io/)
* [UCL dVRK](https://www.ucl.ac.uk/interventional-surgical-sciences/weiss-open-research/weiss-open-data-server/ex-vivo-dvrk-segmentation-dataset-kinematic-data)
* [CholecSeg8k](https://www.kaggle.com/datasets/newslab/cholecseg8k)
* [SAR-RARP50](https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_train_set/24932529)
* [STIR](https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared)

### Data Preparation
Please reformat each dataset according to the following top-level directory layout.

    .
    ├── ...
    ├── train                  
    │   ├── imgs 
    │   │   ├──000000.png   
    │   │   ├──000001.png
    │   │   └──...                 
    │   └── gts
    │       ├──000000.png
    │       ├──000001.png
    │       └──...                
    ├── val
    │    ├── imgs
    │    │   ├──000000.png
    │    │   ├──000001.png
    │    │   └──... 
    │    └── gts
    │        ├──000000.png
    │        ├──000001.png
    │        └──...
    └──...

### Training

Download checkpoints of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM?tab=readme-ov-file), [CoTracker](https://github.com/facebookresearch/co-tracker?tab=readme-ov-file), and [Light HQ-SAM](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth). Put them into `./ckpts` 

If use single GPU for training, run:
```
python train.py -i ./data/[dataset]/train/ -v ./data/[dataset]/val/ --sam-ckpt ./ckpt/mobile_sam.pt --work-dir [path of the training results] --max-epochs 100 --data-aug --freeze-prompt-encoder --batch-size 4 --learn-rate 1e-5 --dataset [dataset]
```
For example:
```
python train.py -i /data/CholecSeg8k/train/ -v /data/CholecSeg8k/val/ --train-from-scratch --work-dir ./results/exp_cholecseg8k --max-epochs 100 --data-aug --freeze-prompt-encoder --batch-size 4 --learn-rate 1e-5 --dataset cholecseg
```
If use multi GPU for training, just add `--multi-gpu` and replace the `device_ids` in line 82 of the `train.py` as the GPU you would like to use:
```
if args.multi_gpu:
    surgicaltool_sam = nn.DataParallel(surgicaltool_sam, device_ids=[0,1,2,3])
```

### Run Online Demo
We need to prepare the first frame and the corresponding mask of the video. If there are more than one tool to segment, please put masks of each tool into a folder. 

Then use `online_demo.py` to run the online demo for a video. 
```
python online_demo.py --video_path [video path] --tracker cotracker --sam_type finetune --tool_number 2 --first_frame_path [path of the first frame of the video] --first_mask_path [path of the first frame mask of the video] --mask_dir_path [folder that contains the mask of each tool in first frame] --save_demo --mode kmedoids --add_support_grid --sam-ckpt ./ckpts/[checkpoint file]
```

## Contact

If you have any problem using this code then create an issue in this repository or contact me at zijianwu@ece.ubc.ca

## License

This project is licensed under the MIT License

## Acknowledgments

Thanks to the following awesome work for the inspiration, code snippets, etc.
* [Segment Anything](https://github.com/facebookresearch/segment-anything)
* [CoTracker](https://github.com/facebookresearch/co-tracker)
* [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
* [HQ-SAM](https://github.com/SysCV/sam-hq)
* [MedSAM](https://github.com/bowang-lab/MedSAM)
* [SAM-PT](https://github.com/SysCV/sam-pt)
