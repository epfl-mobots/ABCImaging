import cv2,os
from tqdm import tqdm

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

def imageHiveOverview(imgs: list,texts: list):
    '''
    Generates a global image with the 4 images of the hives. Also adds the timestamp on the pictures.
    '''
    # put the texts on each image
    for j in range(len(imgs)):
        cv2.putText(imgs[j], texts[j], (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    # Concatenate the images horizontally
    img_top = cv2.hconcat([imgs[0], imgs[2]]) # Frame 1 and 3 on top
    img_bottom = cv2.hconcat([imgs[1], imgs[3]]) # Frame 2 and 4 on bottom
    img = cv2.vconcat([img_top, img_bottom])
    # Resize the image to 4K
    img = cv2.resize(img, (3840, 2160), interpolation=cv2.INTER_LINEAR)
    dt = texts[0].split("_")[-1]
    dt = dt[:-7]
    
    # Write the timestamp in black ontop of the white rectangle
    (text_width, text_height), _ = cv2.getTextSize(dt, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    rectangle_bgr = (255, 255, 255)
    box_coords = ((1700, 1060 + 15), (1700 + text_width, 1060 - text_height - 15))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, dt, (1700, 1060), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    return img