# Augmenting Efficient Surgical Instrument Segmentation in Video with Point Tracking and Segment Anything (SIS-PT-SAM)

PyTorch implementation of the SIS-PT-SAM.

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* Python 3.11
* torch 2.1.2
* torchvision 0.16.2

### Environment Setup

Run the following command to install environment
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

To run `online_demo.py`, we need to prepare the first frame and the corresponding mask of the video. If there are more than one tool to segment, please put masks of each tool into a folder. 

### Training
If use single GPU for training, run:
```
python train.py -i ./data/[dataset name]/train/ -v ./data/[dataset name]/val/ --sam-ckpt ./ckpt/[checkpoint file] --work-dir [path of the training results] --max-epochs 100 --data-aug --freeze-prompt-encoder --batch-size 4 --learn-rate 1e-5 --dataset [dataset]
```

If use multi GPU for training, run:
```
python train.py -i ./data/[dataset name]/train/ -v ./data/[dataset name]/val/ --sam-ckpt ./ckpt/[checkpoint file] --work-dir [path of the training results] --max-epochs 100 --data-aug --freeze-prompt-encoder --batch-size 4 --learn-rate 1e-5 --dataset [dataset] --multi-gpu
```

and replace the `device_ids` in line 82 of the `train.py` as the GPU you would like to use.
```
if args.multi_gpu:
    surgicaltool_sam = nn.DataParallel(surgicaltool_sam, device_ids=[0,1,2,3])
```

### Run Online Demo
Use `online_demo.py` to run the online demo for a video. 
```
python online_demo.py --video_path [video path] --tracker cotracker --sam_type finetune --tool_number 2 --first_frame_path [path of the first frame of the video] --first_mask_path [path of the first frame mask of the video] --mask_dir_path [folder that contains the mask of each tool in first frame] --save_demo --mode kmedoids --add_support_grid --sam-ckpt ./ckpts/[checkpoint file]
```

## Contact

If you have any problem using this code then create an issue in this repository or contact me at zijianwu@ece.ubc.ca

<!-- ## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie) -->

<!-- ## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release -->

## License

This project is licensed under the [NAME HERE] License

## Acknowledgments

Thanks to the following awesome work for the inspiration, code snippets, etc.
* [Segment Anything](https://github.com/facebookresearch/segment-anything)
* [CoTracker](https://github.com/facebookresearch/co-tracker)
* [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
* [HQ-SAM](https://github.com/SysCV/sam-hq)
* [MedSAM](https://github.com/bowang-lab/MedSAM)
* [SAM-PT](https://github.com/SysCV/sam-pt)