import os
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join
import glob
import numpy as np
import random
import cv2
import shutil
from sklearn_extra.cluster import KMedoids 

def min_max_normolization(image):
    """
    image: (3, H, W)
    """
    image = (image - image.min()) / np.clip(image.max() - image.min(), a_min=1e-8, a_max=None)
    return image

def standardization(image, mean=None, std=None):
    """
    image: (3, H, W)
    """
    if mean:
        mean = mean
    else:
        mean = [0.485, 0.456, 0.406] # RGB
    if std:
        std = std
    else:
        std = [0.229, 0.224, 0.225] # RGB
    image[0] = (image[0] - mean[2]) / std[2]
    image[1] = (image[1] - mean[1]) / std[1]
    image[2] = (image[2] - mean[0]) / std[0]
    return image

class FinetuneDataset(Dataset): 
    def __init__(self, data_root, dataset_name, image_size=1024, data_aug=True, status='train', point_num=5):
        self.dataset = dataset_name
        self.status = status
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, '**/*.png'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
        self.image_size = image_size
        self.data_aug = data_aug
        self.point_num = point_num
    
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        assert img_name == os.path.basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]

        img = cv2.imread(join(self.img_path, img_name)) # (H, W, 3) BGR
        img_1024 = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR) # (1024, 1024, 3)
        img_1024 = np.transpose(img_1024, (2, 0, 1)) # (3, 1024, 1024)

        gt = cv2.imread(self.gt_path_files[index], cv2.IMREAD_GRAYSCALE) # (H, W)
        gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        gt = np.uint8(gt)
        assert gt.shape == (1024, 1024)
        gt2D = gt.copy()
        gt2D[gt2D!=0] = 1 # instance mask (gt) -> binary mask (gt2D)\

        # assert label_ids is None, f"gt {self.gt_path_files[index]}, {img_name} is empty"

        # image normalization
        img_1024 = min_max_normolization(img_1024)
        
        # image standardization
        # TODO: calculate mean and std of surgical datasets
        img_1024 = standardization(img_1024)

        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                gt = np.ascontiguousarray(np.flip(gt, axis=-1))
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                gt = np.ascontiguousarray(np.flip(gt, axis=-2))

        # randomly choose prompt at scale 1024
        # In a batch, the number of points should be same...
        if self.dataset == 'endovis17' or self.dataset == 'robustmis19':
            coords = []
            label_ids = np.unique(gt)[1:].tolist()
            point_num = 5
            if np.all(gt==0): # if no object in the image, randomly choose points
                x_indices, y_indices = np.where(gt == 0)
                candidate_points_num = x_indices.shape[0]
                assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                point_idx = np.random.choice(range(candidate_points_num), point_num)
                for idx in point_idx:
                    coords.append([x_indices[idx], y_indices[idx]])
                coords = np.array(coords)
            else:
                for label_id in label_ids:
                    x_indices, y_indices = np.where(gt == label_id)
                    candidate_points_num = x_indices.shape[0]
                    assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                    point_idx = np.random.choice(range(candidate_points_num))
                    x_point, y_point = x_indices[point_idx], y_indices[point_idx]
                    assert gt2D[x_point, y_point] == 1, 'prompt point should be in the mask'
                    coords.append([x_point, y_point])
                if len(label_ids) < (point_num+1):
                    for i in range(point_num - len(label_ids)):
                        x_indices, y_indices = np.where(gt > 0)
                        candidate_points_num = x_indices.shape[0]
                        assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                        point_idx = np.random.choice(range(candidate_points_num))
                        x_point, y_point = x_indices[point_idx], y_indices[point_idx]
                        assert gt2D[x_point, y_point] == 1, 'prompt point should be in the mask'
                        coords.append([x_point, y_point])
                coords = np.array(coords)  
            assert coords.shape == (point_num, 2), 'prompt size should be (point_num, 2)'

        elif self.dataset == 'endovis15':
            coords = []
            point_num = 5
            label_ids = np.unique(gt)[1:].tolist()
            label_id = 1
            # if len(label_ids) > 1:
            #     sub_point_num = random.choice(range(1, point_num))
            #     point_num_list = [sub_point_num, point_num - sub_point_num]
            # else:
            #     point_num_list = [point_num]
            check_points = False
            gt_vis = gt2D.copy()
            gt_vis[gt_vis!=0] = 255
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2RGB)

            x_indices, y_indices = np.where(gt2D == label_id)
            assert x_indices is not None and y_indices is not None, "There's no target object"
            candidate_points_num = x_indices.shape[0]

            assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'

            point_idx = np.random.choice(range(candidate_points_num), point_num)

            for idx in point_idx:
                coords.append([x_indices[idx], y_indices[idx]])
                # print(x_indices[idx], y_indices[idx])
                assert gt2D[x_indices[idx], y_indices[idx]] == label_id, 'prompt point should be in the mask'                    
                # check point prompt
                if check_points:
                    cv2.circle(gt_vis, (y_indices[idx], x_indices[idx]), 3, (0, 0, 255), -1) # Note it's (y,x) not (x,y)
            # cv2.imwrite(f'/home/zijianwu/projects/def-timsbc/zijianwu/results/debug_check/{img_name}.png', gt_vis)
            coords = np.array(coords)  
            assert coords.shape == (point_num, 2), 'prompt size should be (5, 2)'
        
        elif self.dataset == 'ucldvrk' or self.dataset == 'endovis18':
            coords = []
            point_num = 10
            label_ids = [1]
            check_points = False
            gt_vis = gt2D.copy()
            gt_vis[gt_vis!=0] = 255
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2RGB)
            if np.unique(gt).shape[0] == 1:
                label_id = 0
                x_indices, y_indices = np.where(gt == 0)
                candidate_points = x_indices.shape[0]
                assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                point_idx = np.random.choice(range(candidate_points), point_num)
                for idx in point_idx:
                    coords.append([x_indices[idx], y_indices[idx]])
                    # print(x_indices[idx], y_indices[idx])
                    assert gt[x_indices[idx], y_indices[idx]] == label_id, 'prompt point should be in the mask'                    
                    # check point prompt
                    if check_points:
                        cv2.circle(gt_vis, (y_indices[idx], x_indices[idx]), 3, (0, 0, 255), -1)
            else:
                for i, label_id in enumerate(label_ids):
                    x_indices, y_indices = np.where(gt == label_id)
                    candidate_points_num = x_indices.shape[0]
                    assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                    point_idx = np.random.choice(range(candidate_points_num), point_num)

                    for idx in point_idx:
                        coords.append([x_indices[idx], y_indices[idx]])
                        # print(x_indices[idx], y_indices[idx])
                        assert gt[x_indices[idx], y_indices[idx]] == label_id, 'prompt point should be in the mask'                    
                        # check point prompt
                        if check_points:
                            cv2.circle(gt_vis, (y_indices[idx], x_indices[idx]), 3, (0, 0, 255), -1) # Note it's (y,x) not (x,y)
            # cv2.imwrite(f'/home/zijianwu/projects/def-timsbc/zijianwu/results/debug_check/{img_name}.png', gt_vis)
            coords = np.array(coords)  
            assert coords.shape == (point_num, 2), 'prompt size should be (point_num, 2)'

        elif self.dataset == 'cholecseg':
            coords = []
            gt[gt!=0] = 1 
            label_id = 1
            x_indices, y_indices = np.where(gt == label_id)
            point_num = 10
            candidate_points_num = x_indices.shape[0]
            assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
            point_idx = np.random.choice(range(candidate_points_num), point_num)

            for idx in point_idx:
                coords.append([x_indices[idx], y_indices[idx]])
                assert gt2D[x_indices[idx], y_indices[idx]] == label_id, 'prompt point should be in the mask'
                assert gt[x_indices[idx], y_indices[idx]] == label_id, 'prompt point should be in the mask'
                
            coords = np.array(coords) # coords (#label_ids, 2)
            assert coords.shape == (point_num, 2), 'prompt size should be (10, 2)'
        
        elif self.dataset == 'strongsegc': # binary mask, dual-tools scene
            coords = []
            point_num = 10

            check_points = False
            gt_vis = gt2D.copy()
            gt_vis[gt_vis!=0] = 255
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2RGB)
            for i, label_id in enumerate(label_ids):
                x_indices, y_indices = np.where(gt == label_id)
                candidate_points_num = x_indices.shape[0]
                assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                point_idx = np.random.choice(range(candidate_points_num), point_num)

                for idx in point_idx:
                    coords.append([x_indices[idx], y_indices[idx]])
                    # print(x_indices[idx], y_indices[idx])
                    assert gt[x_indices[idx], y_indices[idx]] == label_id, 'prompt point should be in the mask'                    
                    # check point prompt
                    if check_points:
                        cv2.circle(gt_vis, (y_indices[idx], x_indices[idx]), 3, (0, 0, 255), -1) # Note it's (y,x) not (x,y)
            # cv2.imwrite(f'/home/zijianwu/projects/def-timsbc/zijianwu/results/debug_check/{img_name}.png', gt_vis)
            coords = np.array(coords)  
            assert coords.shape == (point_num, 2), 'prompt size should be (10, 2)'

        elif self.dataset == 'sarrarp':
            coords = []
            point_num = 10
            check_points = False

            gt_vis = gt.copy()
            gt_vis[gt_vis==1] = 255 # shaft
            gt_vis[gt_vis==2] = 255 # wrist
            gt_vis[gt_vis==3] = 255 # tip
            gt_vis[gt_vis!=255] = 0
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2RGB)

            gt[gt == 2] = 1
            gt[gt == 3] = 1
            gt[gt != 1] = 0

            gt2D = gt.copy()

            if np.unique(gt).shape[0] != 1:
                label_id = 1
                x_indices, y_indices = np.where(gt == label_id)
                candidate_points = x_indices.shape[0]
                assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                point_idx = np.random.choice(range(candidate_points), point_num)
                for idx in point_idx:
                    coords.append([x_indices[idx], y_indices[idx]])
                    # print(x_indices[idx], y_indices[idx])
                    assert gt[x_indices[idx], y_indices[idx]] == label_id, 'prompt point should be in the mask'                    
                    # check point prompt
                    if check_points:
                        cv2.circle(gt_vis, (y_indices[idx], x_indices[idx]), 3, (0, 0, 255), -1) # Note it's (y,x) not (x,y)
            else: 
                # no robotic instrument in the image
                label_id = 0
                x_indices, y_indices = np.where(gt == 0)
                candidate_points = x_indices.shape[0]
                assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                point_idx = np.random.choice(range(candidate_points), point_num)
                for idx in point_idx:
                    coords.append([x_indices[idx], y_indices[idx]])
                    # print(x_indices[idx], y_indices[idx])
                    assert gt[x_indices[idx], y_indices[idx]] == label_id, 'prompt point should be in the mask'                    
                    # check point prompt
                    if check_points:
                        cv2.circle(gt_vis, (y_indices[idx], x_indices[idx]), 3, (0, 0, 255), -1)
            if check_points:
                os.makedirs('./results/debug_check', exist_ok=True)
                cv2.imwrite(f'./results/debug_check/{img_name}.png', gt_vis)
            coords = np.array(coords)  
            assert coords.shape == (point_num, 2), 'prompt size should be (10, 2)'

        elif self.dataset == 'autolaparo':
            coords = []
            gt[gt==20] = 1
            gt[gt==40] = 1
            gt[gt==60] = 2
            gt[gt==80] = 2 
            gt[gt==100] = 3
            gt[gt==120] = 3 
            gt[gt==140] = 4
            gt[gt==160] = 4
            gt[gt==180] = 0

            label_ids = np.unique(gt)[1:].tolist() 
            point_num = 5

            if np.all(gt==0): # if no object in the image, randomly choose points
                x_indices, y_indices = np.where(gt == 0)
                candidate_points_num = x_indices.shape[0]
                assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                point_idx = np.random.choice(range(candidate_points_num), point_num)
                for idx in point_idx:
                    coords.append([x_indices[idx], y_indices[idx]])
                coords = np.array(coords)
            else:
                for label_id in label_ids:
                    x_indices, y_indices = np.where(gt == label_id)
                    candidate_points_num = x_indices.shape[0]
                    assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                    point_idx = np.random.choice(range(candidate_points_num))
                    x_point, y_point = x_indices[point_idx], y_indices[point_idx]
                    assert gt[x_point, y_point] == label_id, 'prompt point should be in the mask'
                    coords.append([x_point, y_point])
                if len(label_ids) < (point_num + 1):
                    for i in range(point_num - len(label_ids)):
                        x_indices, y_indices = np.where(gt > 0)
                        candidate_points_num = x_indices.shape[0]
                        assert x_indices.shape[0] == y_indices.shape[0], 'length of x_indices and y_indices should be same'
                        point_idx = np.random.choice(range(candidate_points_num))
                        x_point, y_point = x_indices[point_idx], y_indices[point_idx]
                        coords.append([x_point, y_point])
                coords = np.array(coords) # (point_num, 2)   
                assert coords.shape == (point_num, 2), 'prompt size should be (point_num, 2)'

        ## resize gt2D to (256, 256)
        gt2D_256 = cv2.resize(
            gt2D,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        return {
            "image": torch.tensor(img_1024).float(),
            "gt2D": torch.tensor(gt2D_256[None, :,:]).long(),
            "coords": torch.tensor(coords[...]).float(),
            "image_name": img_name
        }
    