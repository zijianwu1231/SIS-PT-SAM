import os
import cv2
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='images', help='path to the folder containing images')
    parser.add_argument('--fps', type=int, default=25, help='frames per second')
    parser.add_argument('--output', type=str, default='output.mp4', help='path to the output video file')

    args = parser.parse_args()
    
    # Get the path to the folder containing images
    path = args.path

    # Get the frames per second
    fps = args.fps

    # Get the path to the output video file
    output = args.output

    # Get the list of all the images
    images_list = sorted(os.listdir(path))#, key=lambda x: int(x.split('.')[0]))
    images = [os.path.join(path, image) for image in images_list]

    # Get the height and width of the first image
    height, width, _ = cv2.imread(images[0]).shape

    # Define the codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Define the video writer
    video = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # Iterate through the images
    for image in images:
        # Read the image
        img = cv2.imread(image)

        # Write the image to the video file
        video.write(img)

    # Release the video writer
    video.release()
