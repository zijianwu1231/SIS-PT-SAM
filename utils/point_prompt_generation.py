import os
import numpy as np
import cv2
import torch
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from typing import Union, Optional, List

def get_KMedoids_centers(mask_path:str, subsample_size:int=1800, n_points_to_select:int=5):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask / 255
    mask_pixels = torch.tensor(mask).nonzero().float()
    mask_pixels = mask_pixels[torch.randperm(len(mask_pixels))[:subsample_size]]
    selected_points = KMedoids(n_clusters=n_points_to_select).fit(mask_pixels).cluster_centers_
    selected_points = torch.from_numpy(selected_points).type(torch.float32)
    selected_points = selected_points.flip(1)
    return selected_points

def get_sift_keypoints(image_path:str, mask_path:str):
    first_frame = cv2.imread(image_path)
    first_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # image_roi = cv2.bitwise_and(first_mask, first_frame, mask=None)
    image_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    sift_detector= cv2.SIFT_create()
    keypoints = sift_detector.detect(image_gray)
    for kpt in keypoints:
        x, y = kpt.pt
        if first_mask[int(y), int(x)] == 255:
            cv2.circle(first_frame, (int(x), int(y)), 5, (0,255,0), -1)
    # output_image = cv2.drawKeypoints(first_frame, keypoints, 0, (255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) 
    plt.imshow(first_frame) 
    plt.show()
    cv2.imwrite('./CoTracker2/assets/sift_keypoints.png', first_frame)
    return keypoints

def get_shi_tomasi_keypoints(image_path:str, mask_path:str):
    first_frame = cv2.imread(image_path)
    first_mask = cv2.imread(mask_path)
    # gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    image_roi = cv2.bitwise_and(first_mask, first_frame, mask=None)
    # breakpoint()
    image_roi_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(image_roi_gray, 10, 0.01, 30) #(25, 0.01, 10)
    corners = np.int0(corners)
    queries = []
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    for i in corners:
        x,y = i.ravel()
        queries.append([0., x, y])
        cv2.circle(first_frame, (x,y), 5, (0,255,0), -1)
    plt.imshow(first_frame),
    plt.axis('off') 
    plt.savefig(os.path.join('./CoTracker2/assets/shi-tomasi.png'), bbox_inches='tight', pad_inches=0)
    plt.show()
    queries = torch.tensor(queries)
    if torch.cuda.is_available():
        queries = queries.cuda()

    return queries

def draw_dashed_contour(segm_mask, first_frame_image):
    # draw the dashed line contour
    contours, _ = cv2.findContours(segm_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    frame_contour = first_frame_image.copy()
    frame_contour = np.array(frame_contour)
    frame_contour = cv2.cvtColor(frame_contour, cv2.COLOR_RGB2BGR)
    cv2.drawContours(frame_contour, contours, -1, (0, 255, 0), 2)
    # plt.imshow(frame_contour)
    # plt.show()
    cv2.imwrite('./CoTracker2/assets/endovis15_clipseg_contour.png', frame_contour)
    # plt.imshow(segm_mask)
    # plt.show()

class PointPromptGenerator:
    def __init__(
        self, 
        image_path:str, 
        mask_path:str, 
        mode:str='kmedoids',
        vis_path:str='../results/query_points', 
        tool_number:int=None,
        mask_dir_path:str=None
    ) -> None:
        self.mode = mode
        self.image_path = image_path
        self.mask_path = mask_path
        self.tool_number = tool_number
        self.vis_image_path = vis_path
        os.makedirs(self.vis_image_path, exist_ok=True)
        self.mask_dir_path = mask_dir_path

    def sample(self, point_num:int) -> list:
        first_frame = cv2.imread(self.image_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        if self.mode == 'customized':
            pass
        elif self.mode == 'random':
            pass
        elif self.mode == 'kmedoids':
            if not self.tool_number or self.tool_number == 1:
                selected_points = get_KMedoids_centers(self.mask_path, n_points_to_select=5)
                queries = []
                for i in selected_points:
                    x, y = i.ravel()
                    queries.append([0., x.item(), y.item()])
                    cv2.circle(first_frame, (int(x.item()), int(y.item())), 3, (0,0,255), -1)
            else: 
                # dual-tool case as an example
                mask_list = os.listdir(self.mask_dir_path)
                mask_path_list = [os.path.join(self.mask_dir_path, mask_name) for mask_name in mask_list]
                queries = []
                for i in range(self.tool_number):
                    selected_points = get_KMedoids_centers(mask_path_list[i], n_points_to_select=point_num)
                    for i in selected_points:
                        x, y = i.ravel()
                        queries.append([0., x.item(), y.item()])
                        cv2.circle(first_frame, (int(x.item()), int(y.item())), 3, (0,0,255), -1)  
            
            # visualize the selected points
            plt.imshow(first_frame)
            plt.axis('off') 
            plt.savefig(os.path.join(self.vis_image_path,'k-medoids.png'), bbox_inches='tight', pad_inches=0)
            plt.show()
            breakpoint()
            plt.close()
            # queries.append([0., 676., 335.]) # if need background point prompt
            return queries

            # queries = torch.tensor(queries)
            # if torch.cuda.is_available():
            #     queries = queries.cuda()          

class PointPromptGeneratorCLIPSeg:
    def __init__(
        self, 
        image_path:str, 
        mode:str='kmedoids', 
        vis_path:str='../results/query_points',
        tool_number:int=None
    ) -> None:
        self.mode = mode
        self.image_path = image_path
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.vis_image_path = vis_path
        os.makedirs(self.vis_image_path, exist_ok=True)

    def segmentation(
        self,
        image_path:str,
        text_prompt:List[str]=["surgical tool", "background"], 
        threshold:float=0.2
    ) -> np.ndarray:
        from PIL import Image
        clipseg_img = Image.open(image_path).convert("RGB")
        (W, H) = clipseg_img.size
        clipseg_inputs = self.clipseg_processor(text=text_prompt, images=[clipseg_img] * len(text_prompt), padding="max_length", return_tensors="pt")
        with torch.no_grad():
            clipseg_outputs = self.clipseg_model(**clipseg_inputs)
        clipseg_preds = clipseg_outputs.logits.unsqueeze(1)
        mask = clipseg_preds[0].squeeze()
        mask = torch.sigmoid(mask)
        mask = cv2.resize(mask.cpu().numpy(), (W, H))
        # plt.imshow(mask)
        # plt.show()
        # breakpoint()
        mask[mask > threshold] = 255
        mask[mask != 255] = 0
        cv2.imwrite(os.path.join(self.vis_image_path, 'CLIPSeg_mask.png'), mask)
        # mask_3c = cv2.merge([mask, mask, mask])
        # plt.figure(figsize=(W, H))
        # plt.imshow(mask_3c)
        # plt.axis('off') 
        # plt.savefig(os.path.join(self.vis_image_path,'CLIPSeg_mask.png'), bbox_inches='tight', pad_inches=0)
        # plt.show()
        # breakpoint()
        # plt.close()
        return mask

    def sample(
        self, 
        point_num:int, 
        seg_threshold:float=0.2,
        text_prompt:List[str]=["surgical tool", "background"]
    ) -> list:
        _ = self.segmentation(self.image_path, text_prompt=text_prompt, threshold=seg_threshold)
        mask_path = os.path.join(self.vis_image_path, 'CLIPSeg_mask.png')
        first_frame = cv2.imread(self.image_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        if self.mode == 'customized':
            pass
        elif self.mode == 'random':
            pass
        elif self.mode == 'kmedoids':
            selected_points = get_KMedoids_centers(mask_path, n_points_to_select=point_num)
            queries = []
            for i in selected_points:
                x, y = i.ravel()
                queries.append([0., x.item(), y.item()])
                cv2.circle(first_frame, (int(x.item()), int(y.item())), 3, (0,0,255), -1)  
                
            plt.imshow(first_frame)
            plt.axis('off') 
            plt.savefig(os.path.join(self.vis_image_path,'k-medoids.png'), bbox_inches='tight', pad_inches=0)
            plt.show()
            breakpoint()
            plt.close()

            return queries
        
def main():
    # test the point prompt generation
    input_mask_path = '../CoTracker2/assets/endovis15_mask_sam_4.png'
    input_image_path = '../CoTracker2/assets/endovis15_img_4.png'
    query_generator = PointPromptGenerator(
        image_path=input_image_path,
        mask_path=input_mask_path,
        mode='kmedoids',
    )
    queries = query_generator.sample(point_num=5)

    # test the point prompt generation based on CLIPSeg
    query_generator_clipseg = PointPromptGeneratorCLIPSeg(
        image_path=input_image_path,
        mode='kmedoids',
    )
    queries_clipseg = query_generator_clipseg.sample(point_num=5, seg_threshold=0.002, text_prompt=["large needle driver", "background"])

if __name__ == '__main__': 
    main()