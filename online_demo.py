# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from torchvision.transforms import ToTensor
import argparse
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
from cotracker.predictor import CoTrackerOnlinePredictor
from sklearn_extra.cluster import KMedoids
from thop import profile
from utils.point_prompt_generation import PointPromptGenerator, PointPromptGeneratorCLIPSeg
# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future
from dataset import min_max_normolization, standardization

DEFAULT_DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

def select_prompt_points(tracks, visibility):
    # tracks: (N, 2), visibility: (N,)
    # return: (M, 2)
    assert tracks.shape[0] == visibility.shape[0], 'the number of points of the tracks and visibility should be the same'
    # select the points with visibility is True
    selected_tracks = tracks[visibility]
    return selected_tracks

def initialize_sam_model(sam_type):
    # Initialize a SAM model
    if sam_type == 'mobile_sam':
        model_type = "vit_t"
        sam_checkpoint = "./ckpts/mobile_sam.pt"
        from mobile_sam import sam_model_registry, SamPredictor
    elif sam_type == 'hq_sam':
        model_type = "vit_tiny"
        sam_checkpoint = "./ckpts/sam_hq_vit_tiny.pth"
        from segment_anything_hq import sam_model_registry, SamPredictor
    elif sam_type == 'sam':
        model_type = "vit_h"
        sam_checkpoint = "./ckpts/sam_vit_h_4b8939.pth" #"./ckpts/medsam_point_prompt_best.pth"
        from segment_anything import sam_model_registry, SamPredictor  
    elif sam_type == 'finetune':
        model_type = "vit_t"
        mobile_sam_checkpoint = './ckpts/mobile_sam.pt'
        # sam_checkpoint = torch.load('./ckpts/surgicaltoolsam_best_ff.pth')
        sam_checkpoint = torch.load(args.sam_ckpt)
        from surgical_tool_sam import SurgicalToolSAM

    if sam_type != "finetune":
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam_model.to(DEFAULT_DEVICE)
        sam_model.eval()
        predictor = SamPredictor(sam_model)
        return sam_model, predictor
    else:
        sam_model = SurgicalToolSAM(
            ckpt=mobile_sam_checkpoint, 
            freeze_image_encoder=True, 
            freeze_prompt_encoder=True, 
            freeze_mask_decoder=True,
        )
        sam_model.load_state_dict(sam_checkpoint['model'])
        sam_model.to(DEFAULT_DEVICE)
        sam_model.eval()
        return sam_model

def point_tracking_step(window_frames, is_first_step, grid_size, add_support_grid, queries=None):
    video_chunk = (
        torch.tensor(np.stack(window_frames[-pt_model.step * 2 :]), device=DEFAULT_DEVICE)
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)
    iter_start_time = time.time()
    pred_tracks, pred_visibility = pt_model(
        video_chunk,
        is_first_step=is_first_step,
        grid_size=grid_size,
        queries=queries[None],
        add_support_grid=add_support_grid
    )
    iter_time = time.time()-iter_start_time
    print('cotracker inference %.1f fps' % (video_chunk.shape[1]/iter_time))
    return pred_tracks, pred_visibility

def visualize_mask(mask_img, vis_folder_path, img_name):
    mask_r = mask_img * 0
    mask_g = mask_img * 0
    mask_b = mask_img * 255
    mask_img = np.stack([mask_r, mask_g, mask_b], axis=2)

    # blend the mask and the frame using cv2.addWeighted()
    blended_img = cv2.addWeighted(mask_img, 0.9, frame, 1, 0)

    # draw points on the frame
    fig = plt.figure()
    plt.imshow(blended_img)
    plt.plot(point_coords[:, 0], point_coords[:, 1], 'go', markersize=3, linewidth=1.0) #,mfc='none', 

    # save the frame
    vis_folder = vis_folder_path 
    os.makedirs(vis_folder, exist_ok=True)
    plt.axis('off') 
    plt.savefig(os.path.join(vis_folder, img_name), bbox_inches='tight', pad_inches=0)
    # plt.show()
    # breakpoint()
    plt.close()    

def frame_input_for_sam(frame_input):
    # frame_input = frame.copy()
    frame_input = cv2.resize(frame_input, (1024, 1024)) # resize (H, W, 3) to (1024, 1024, 3)
    frame_input = np.uint8(frame_input)  
    # frame_input = (frame_input - frame_input.min()) / np.clip(frame_input.max() - frame_input.min(), a_min=1e-8, a_max=None)
    frame_input = torch.from_numpy(frame_input)
    frame_input = frame_input.permute(2, 0, 1)#.float()[None,:] # (B, C, H, W) (1, 3, 1024, 1024)
    frame_input = min_max_normolization(frame_input)
    frame_input = standardization(frame_input)
    frame_input = frame_input.float()[None, :]
    frame_input = frame_input.to(DEFAULT_DEVICE)
    return frame_input

def point_embeddding_input_for_sam(point_input):
    point_input[:, 0] = (1024 / W) * point_input[:, 0]
    point_input[:, 1] = (1024 / H) * point_input[:, 1]    
    point_coords_input = torch.from_numpy(point_input.astype(int)[None, :])
    point_labels_input = torch.from_numpy(point_labels[None, :])
    point_coords_input = point_coords_input.to(DEFAULT_DEVICE)
    point_labels_input = point_labels_input.to(DEFAULT_DEVICE)
    point_prompt_input = (point_coords_input, point_labels_input) # (B, N, 2), (B, N)
    return point_prompt_input

def draw_dashed_line_contour(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    frame_contour = image.copy()
    frame_contour = cv2.cvtColor(frame_contour, cv2.COLOR_RGB2BGR)
    cv2.drawContours(frame_contour, contours, -1, (0, 255, 0), 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="path to a video",)
    parser.add_argument("--tracker", "-t", default=str, help="cotracker or pips2",)
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="Compute dense and grid tracks starting from this frame",)
    parser.add_argument("--mode", "-m", type=str, default="global_girds", help="type of point prompts",)
    parser.add_argument("--sam_type", "-v", type=str, default="mobile_sam", help="type of the SAM model, now only support mobile_sam, hq_sam, and sam",)
    parser.add_argument("--tool_number", type=int, default=None, help="the number of tool in the video",)
    parser.add_argument("--instance_point_number", type=int, default=5, help="the number of point prompts of each instance",)
    parser.add_argument("--add_support_grid", action="store_true", help="if add support grids",)
    parser.add_argument("--use_clipseg", type=bool, default=False, help="if use clipseg to provide first frame mask",)
    parser.add_argument("--output_mask_path", "-out", type=str, default="./results/predicted_mask", help="path to the output binary mask",)
    parser.add_argument("--output_vis_path", "-vis", type=str, default="./results/vis_mask", help="path to the output visualization mask",)
    
    parser.add_argument("--first_frame_path", "-ff", type=str, default="./data/assets/endovis15_img_1.png", help="path to the first frame",)
    parser.add_argument("--first_mask_path", "-fm", type=str, default="./data/assets/endovis15_mask_sam_1.png", help="path to the mask of the first frame",)
    parser.add_argument("--mask_dir_path", "-md", type=str, default="./data/assets/endovis15_test_1", help="path to the mask directory",)
    parser.add_argument("--save_demo", action="store_true", help="if saving demo video",)
    parser.add_argument("--sam-ckpt", type=str, help="sam ckpt path",)

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    # Initialize CoTracker
    if args.tracker == "cotracker":
        tracker_ckpt = "./ckpts/cotracker2.pth"
        pt_model = CoTrackerOnlinePredictor(checkpoint=tracker_ckpt)
        pt_model = pt_model.to(DEFAULT_DEVICE)
    else:
        pt_model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
        pt_model = pt_model.to(DEFAULT_DEVICE)

    # Load the first frame and the mask
    first_frame_path = args.first_frame_path 
    input_mask_path = args.first_mask_path
    if args.tool_number:
        mask_dir_path = args.mask_dir_path
    else:
        mask_dir_path = None
    segm_mask = np.array(Image.open(input_mask_path))
    if segm_mask.shape[-1] == 3:
        segm_mask = cv2.cvtColor(segm_mask, cv2.COLOR_BGR2GRAY)
    
    # Initialize SAM model
    if args.sam_type == "finetune":
        sam_model = initialize_sam_model(args.sam_type)
    else:
        sam_model, predictor = initialize_sam_model(args.sam_type)     
    
    # Pick some points as queries for the point tracker:
    if args.mode == "customized":
        queries = torch.tensor([
            [0., 420., 270.],
            [0., 427., 263.],
            [0., 475., 278.],
            [0., 630., 238.],
            [0., 682., 219.]
        ])
    elif args.mode == "kmedoids":
        if args.use_clipseg:
            print('###########using CLIPSeg#############')
            query_generator_clipseg = PointPromptGeneratorCLIPSeg(
                image_path=first_frame_path,
                mode='kmedoids',
                vis_path='./results/query_points',
                tool_number=args.tool_number
            )
            queries = query_generator_clipseg.sample(point_num=5, seg_threshold=0.002, text_prompt=["large needle driver", "background"])
        else:
            query_generator = PointPromptGenerator(
                image_path=first_frame_path,
                mask_path=input_mask_path,
                mode='kmedoids',
                vis_path='./results/query_points',
                tool_number=args.tool_number,
                mask_dir_path=mask_dir_path,
            )
            queries = query_generator.sample(point_num=5)
        queries = torch.tensor(queries)

    if torch.cuda.is_available():
        queries = queries.cuda()

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    track_labels = None
    window_frames = []

    # Start tracking    
    total_time = 0
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        if i % pt_model.step == 0 and i != 0:
            pred_tracks, pred_visibility = point_tracking_step(
                window_frames,
                is_first_step,
                grid_size = 0,
                queries=queries,
                add_support_grid=args.add_support_grid
            )
            if is_first_step:
                # number of the query points
                queries_num_init = pt_model.queries.shape[1] # queries shape (1, N, 3)
                print(f'Initial # of query points: {queries_num_init}')

            if pred_tracks is not None:
                assert pred_tracks.shape[1] == len(window_frames), 'every frame should have a prediction'
                ## SAM predict and draw points on the frame
                if i == 8: # process the first 8 frames (default video chunk length is 4 frames)
                    for j in range(2*pt_model.step, 0, -1):
                        tracks = pred_tracks[:,-j,:,:].detach().cpu().numpy().squeeze()
                        visibility = pred_visibility[:,-j,:].detach().cpu().numpy().squeeze()
                        frame = window_frames[-j] # (H, W, 3)

                        # assign positive and negative labels to the tracks
                        if j == 8:
                            track_labels = np.array([segm_mask[int(tracks[k, 1]), int(tracks[k, 0])] for k in range(tracks.shape[0])]) == 255
                            if args.tool_number is None or args.tool_number == 1:
                                track_labels[args.instance_point_number:] = [False for ii in track_labels[args.instance_point_number:]]# track_labels[5:] = [False for ii in track_labels[5:]]
                            else:
                                track_labels[(args.instance_point_number*args.tool_number):] = [False for ii in track_labels[(args.instance_point_number*args.tool_number):]]

                            selected_tracks = select_prompt_points(tracks, np.logical_and(visibility, track_labels))
                            point_coords = selected_tracks
                            point_labels = np.ones(len(point_coords), dtype=np.uint8)
                            
                        assert point_coords.shape[0] == point_labels.shape[0], 'the number of prompt points and labels should be the same'

                        # SAM predict
                        if args.sam_type != "finetune":
                            start_time = time.time()
                            predictor.set_image(frame)
                            masks, scores, _ = predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                            )
                            iter_time = time.time() - start_time
                            total_time += iter_time
                            max_score = scores.max()
                            max_idx = np.where(scores == max_score)
                            mask = masks[max_idx]
                            mask_img = mask.astype(np.uint8).squeeze()

                            mask_out = mask_img.copy()
                            mask_out[mask_out > 0] = 255
                            os.makedirs(args.output_mask_path, exist_ok=True)
                            mask_out_name = os.path.join(args.output_mask_path, 'mask_'+str(i-j).zfill(4)+'.png')
                            cv2.imwrite(mask_out_name, mask_out)
                        else:
                            (H, W, C) = frame.shape

                            point_input = point_coords.copy()
                            point_prompt_input = point_embeddding_input_for_sam(point_input)                          
                            frame_input = frame.copy()
                            frame_input = frame_input_for_sam(frame_input)   

                            start_time = time.time()
                            mask = sam_model(frame_input, point_prompt_input, 'inference')
                            iter_time = time.time()-start_time
                            total_time += iter_time

                            mask = cv2.resize(mask, (W, H))
                            mask_img = mask.astype(np.uint8).squeeze()
                            mask_out = mask_img.copy()
                            mask_out[mask_out > 0] = 255
                            os.makedirs(args.output_mask_path, exist_ok=True)
                            mask_out_name = os.path.join(args.output_mask_path, 'mask_'+str(i-j).zfill(4)+'.png')
                            cv2.imwrite(mask_out_name, mask_out)
                            # draw the dashed line contour
                            # draw_dashed_line_contour(frame, mask_out)

                        vis_img_name = 'frame_' + str(i-j).zfill(4) + '.png'
                        os.makedirs(args.output_vis_path, exist_ok=True)
                        visualize_mask(mask_img, args.output_vis_path, vis_img_name)

                else: # process the rest frames
                    for j in range(pt_model.step, 0, -1):
                        tracks = pred_tracks[:,-j,:,:].detach().cpu().numpy().squeeze()
                        visibility = pred_visibility[:,-j,:].detach().cpu().numpy().squeeze()
                        frame = window_frames[-j]
                        # print(track_labels)
                        selected_tracks = select_prompt_points(tracks, np.logical_and(visibility, track_labels))
                        point_coords = selected_tracks
                        point_labels = np.ones(len(point_coords), dtype=np.uint8)
                            
                        if args.sam_type != "finetune":
                            start_time = time.time()
                            predictor.set_image(frame)
                            masks, scores, _ = predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                            )
                            iter_time = time.time()-start_time
                            total_time += iter_time
                            print(f'SAM image encoder inference speed {iter_time} seconds')
                            max_score = scores.max()
                            max_idx = np.where(scores == max_score)
                            mask = masks[max_idx]
                            mask_img = mask.astype(np.uint8).squeeze()

                            mask_out = mask_img.copy()
                            mask_out[mask_out > 0] = 255
                            mask_out_name = os.path.join(args.output_mask_path, 'mask_'+str(i-j).zfill(4)+'.png')
                            cv2.imwrite(mask_out_name, mask_out)                            
                        else:
                            (H, W, C) = frame.shape
                            point_input = point_coords.copy()
                            point_prompt_input = point_embeddding_input_for_sam(point_input)
                            frame_input = frame.copy()
                            frame_input = frame_input_for_sam(frame_input)

                            start_time = time.time()
                            mask = sam_model(frame_input, point_prompt_input, 'inference')
                            iter_time = time.time()-start_time
                            total_time += iter_time

                            mask = cv2.resize(mask, (W, H))
                            mask_img = mask.astype(np.uint8).squeeze()
                            mask_out = mask_img.copy()
                            mask_out[mask_out > 0] = 255
                            mask_out_name = os.path.join(args.output_mask_path, 'mask_'+str(i-j).zfill(4)+'.png')
                            cv2.imwrite(mask_out_name, mask_out)      
                            # draw the dashed line contour
                            # draw_dashed_line_contour(frame, mask_out)         

                        vis_img_name = 'frame_' + str(i-j).zfill(4) + '.png'
                        visualize_mask(mask_img, args.output_vis_path, vis_img_name)

            is_first_step = False
        window_frames.append(frame)

    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = point_tracking_step(
        window_frames[-(i % pt_model.step) - pt_model.step - 1 :],
        is_first_step,
        grid_size = 0,
        queries=queries,
        add_support_grid=args.add_support_grid
    )
    for k in range(((i+1)%pt_model.step), 0, -1):
        tracks = pred_tracks[:,-k,:,:].detach().cpu().numpy().squeeze()
        visibility = pred_visibility[:,-k,:].detach().cpu().numpy().squeeze()
        frame = window_frames[-k]
        # print(track_labels)
        selected_tracks = select_prompt_points(tracks, np.logical_and(visibility, track_labels))
        point_coords = selected_tracks
        point_labels = np.ones(len(point_coords), dtype=np.uint8)
            
        if args.sam_type != "finetune":
            start_time = time.time()
            predictor.set_image(frame)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
            )
            iter_time = time.time()-start_time
            total_time += iter_time
            print(f'SAM image encoder inference speed {iter_time} seconds')
            max_score = scores.max()
            max_idx = np.where(scores == max_score)
            mask = masks[max_idx]
            mask_img = mask.astype(np.uint8).squeeze()

            mask_out = mask_img.copy()
            mask_out[mask_out > 0] = 255
            mask_out_name = os.path.join(args.output_mask_path, 'mask_'+str(i+1-k).zfill(4)+'.png')
            cv2.imwrite(mask_out_name, mask_out)                            
        else:
            (H, W, C) = frame.shape
            point_input = point_coords.copy()
            point_prompt_input = point_embeddding_input_for_sam(point_input)
            frame_input = frame.copy()
            frame_input = frame_input_for_sam(frame_input)

            start_time = time.time()
            mask = sam_model(frame_input, point_prompt_input, 'inference')
            iter_time = time.time()-start_time
            total_time += iter_time

            mask = cv2.resize(mask, (W, H))
            mask_img = mask.astype(np.uint8).squeeze()
            mask_out = mask_img.copy()
            mask_out[mask_out > 0] = 255
            mask_out_name = os.path.join(args.output_mask_path, 'mask_'+str(i+1-k).zfill(4)+'.png')
            cv2.imwrite(mask_out_name, mask_out)           

        vis_img_name = 'frame_' + str(i+1-k).zfill(4) + '.png'
        visualize_mask(mask_img, args.output_vis_path, vis_img_name)
    
    print("All frames are processed")
    print(f"Average SAM inference speed: {1 / (total_time/len(window_frames))} FPS")

    # Save a demo video
    if args.save_demo:
        demo_video_path = os.path.join('./results', 'demo.mp4')
        demo_frames_path = args.output_vis_path
        images = sorted([os.path.join(demo_frames_path, image) for image in os.listdir(demo_frames_path)])
        height, width, _ = cv2.imread(images[0]).shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Define the video writer
        fps = 25
        video = cv2.VideoWriter(demo_video_path, fourcc, fps, (width, height))
        # Iterate through the images
        for image in images:
            # Read the image
            img = cv2.imread(image)
            # Write the image to the video file
            video.write(img)
        # Release the video writer
        video.release()