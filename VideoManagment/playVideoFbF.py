'''
This script plays a video frame by frame from an .mp4 (or other) video file.
'''

import cv2

video_path = '/Users/cyrilmonette/Desktop/EPFL 2018-2026/PhD - Mobots/data/22.12-23.02_actuation_OH/Images/high-fps/h5r2_22.12/hive5_rpi2_day-221228.mp4'


def play_video(video_path):
    # load video capture from file
    video = cv2.VideoCapture(video_path)
    # window name and size
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    while video.isOpened():
        # Read video capture
        ret, frame = video.read()
        # Display each frame
        cv2.imshow("video", frame)
        # show one frame at a time
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0)
        # Quit when 'q' is pressed
        if key == ord('q'):
            break
        if key==ord('j'):
            # Go back one frame
            video.set(cv2.CAP_PROP_POS_FRAMES, video.get(cv2.CAP_PROP_POS_FRAMES) - 2)
            
    # Release capture object
    video.release()
    # Exit and distroy all windows
    cv2.destroyAllWindows()


play_video(video_path)