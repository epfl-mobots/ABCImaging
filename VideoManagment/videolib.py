import cv2,os
from tqdm import tqdm
import pandas as pd

def generateVideoFromDir():
    '''
    This function generates a video from a sequence of pictures in a directory.
    '''

def generateVideoFromList(imgs:list, dest, name:str="video", fps:int=10,grayscale=True):
    '''
    This function generates a video from a list of images.
    '''
    # Checks on the inputs
    if not os.path.isdir(dest):
        raise ValueError("dest must be a valid directory")
    if len(imgs) == 0:
        raise ValueError("imgs must be a non-empty list")

    # Generate mp4 video from final_imgs
    if grayscale:
        height, width = imgs[0].shape
    else:
        height, width, _ = imgs[0].shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    # Convert destination from PosixPath to string
    name = str(dest)+'/'+name+".mp4"
    video = cv2.VideoWriter(name, fourcc, fps, size,isColor=not grayscale)

    # Iterate over the frames and write each one to the video
    print("Writing video...")
    for frame in tqdm(imgs):
        video.write(frame)

    # Release the VideoWriter object
    video.release()

def initVideoWriter(dest, frame, name:str="video",fps:int=10, grayscale=True):
    '''
    This function initializes a VideoWriter object to write frames to a video file.
    If the video file already exists, it will be overwritten.
    The frame param is used to determine the size of the video.
    '''
    # Checks on the inputs
    if not os.path.isdir(dest):
        raise ValueError("dest must be a valid directory")
    
    # Generate mp4 video from final_imgs
    if grayscale:
        height, width = frame.shape
    else:
        height, width, _ = frame.shape
    size = (width, height)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    # Convert destination from PosixPath to string
    name = str(dest)+'/'+name+".mp4"
    video = cv2.VideoWriter(name, fourcc, fps, size,isColor=not grayscale)
    return video


def imageHiveOverview(imgs: list, img_names: list[str]= None, dt: pd.Timestamp = None, valid: bool = True):
    '''
    Generates a global image with the 4 images of the hives. If provided, also adds the img_names on the pictures.
    Parameters:
    - imgs: list of 4 images (numpy arrays) to be concatenated.
    - img_names: list of str, optional, names of the images to be put on the images.
    - dt: optional, datetime to be displayed on the final image.
    - valid: bool, if False, the overview will be visually marked as invalid.
    '''
    if img_names is not None:
        # put the img names on each image
        for j in range(len(imgs)):
            cv2.putText(imgs[j], img_names[j], (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    # Concatenate the images horizontally
    img_top = cv2.hconcat([imgs[0], imgs[2]]) # Frame 1 and 3 on top
    img_bottom = cv2.hconcat([imgs[1], imgs[3]]) # Frame 2 and 4 on bottom
    img = cv2.vconcat([img_top, img_bottom])
    # Resize the image to 4K
    img = cv2.resize(img, (3840, 2160), interpolation=cv2.INTER_LINEAR)

    if dt is not None:
        # Make sure it is tz-aware
        assert dt.tzinfo is not None, "dt must be tz-aware"
        dt = dt.tz_convert('UTC').strftime("%y%m%d-%H%M") + "Z"  # Convert to UTC and format as string
        # Write the timestamp in black ontop of the white rectangle
        (text_width, text_height), _ = cv2.getTextSize(dt, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        rectangle_bgr = (255, 255, 255)
        box_coords = ((1700, 1060 + 15), (1700 + text_width, 1060 - text_height - 15))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img, dt, (1700, 1060), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    
    if not valid:
        # Convert img to BGR if it is grayscale
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # If the image is not valid, put a transparent filter on the image
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (3840, 2160), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        # Add "Invalid" text on the image
        cv2.putText(img, "Invalid dt", (1500, 500), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 15, cv2.LINE_AA)
        cv2.putText(img, "Invalid dt", (1500, 1600), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 15, cv2.LINE_AA)

    return img