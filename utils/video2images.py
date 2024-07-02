import os
import cv2
import argparse

if __name__ == "__main__":
        
        parser = argparse.ArgumentParser()
    
        parser.add_argument('--video-path', type=str, default='images', help='path to the input video file')
        parser.add_argument('--fps', type=int, default=25, help='frames per second')
        parser.add_argument('--output', type=str, help='path to the folder containing output images')
        parser.add_argument('--start-frame', type=int, default=0, help='start frame')
        parser.add_argument('--end-frame', type=int, default=None, help='end frame')
    
        args = parser.parse_args()
        
        # Get the path to the video file
        path = args.video_path
    
        # Get the frames per second
        fps = args.fps
    
        # Get the path to the output video file
        output = args.output
    
        # video to images using opencv
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 250
        while success:
            if count >= args.start_frame:
                cv2.imwrite(os.path.join(output, f"{str(count).zfill(9)}.png"), image)     # save frame as JPEG file      
            success, image = vidcap.read()
            count += 1
            if count == args.end_frame:
                break
        
        print("Total frames: ", count)
        print("Done!")