import os
from sklearn.metrics import jaccard_score, f1_score, precision_recall_fscore_support
import cv2
import matplotlib.pyplot as plt
import argparse

def calculate_metrics(pred_path, gt_path):
    preds = sorted(os.listdir(pred_path))
    gts  = sorted(os.listdir(gt_path))
    iou_total, dice_total = 0, 0 # J & F

    for gt, pred in zip(gts, preds):
        gt_image = cv2.imread(os.path.join(gt_path, gt), cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(os.path.join(pred_path, pred), cv2.IMREAD_GRAYSCALE)

        gt_image[gt_image != 0] = 255
        pred_image[pred_image > 0] = 255

        gt_image = gt_image / 255
        pred_image = pred_image / 255
        
        iou = jaccard_score(gt_image.flatten(), pred_image.flatten()) # Jaccard
        dice = f1_score(gt_image.flatten(), pred_image.flatten()) # F1 score

        iou_total += iou
        dice_total += dice
        # p,r,_,_ = precision_recall_fscore_support(gt_image.flatten(), pred_image.flatten())
        # f1_ = 2*p*r/(p+r)
        # f_total += f1_

    avrg_iou = iou_total / len(gts)
    avrg_dice = dice_total / len(gts)

    return avrg_iou, avrg_dice

if __name__ == '__main__':
    print('start evaluation...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default='./results/predicted_mask', help='path to predicted masks')
    parser.add_argument('--gt_path', type=str, help='path to ground truth masks')
    args = parser.parse_args()

    pred_path = args.pred_path
    gt_path = args.gt_path
    # pred_path = '/data/SurgicalToolTracking/results/predicted_mask'
    # gt_path = '/data/CholecSeg8k/val/video1/masks'

    avrg_iou, avrg_dice = calculate_metrics(pred_path, gt_path)

    print(avrg_iou)
    print(avrg_dice)
