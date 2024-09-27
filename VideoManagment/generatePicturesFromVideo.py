import cv2
import pandas as pd

# Load the video
video_path = '/Users/cyrilmonette/Desktop/EPFL 2018-2026/PhD - Mobots/data/22.12-23.02_actuation_OH/Images/high-fps/h5r2_22.12/hive5_rpi2_day-221228.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Oops! We couldn't open the video.")

# Load csv file with timestamps
csv_path = '/Users/cyrilmonette/Desktop/EPFL 2018-2026/PhD - Mobots/data/22.12-23.02_actuation_OH/Images/high-fps/h5r2_22.12/hive5_rpi2_day-221228.csv'
df = pd.read_csv(csv_path,header=1)
print(df.head())   

frame_count = 0
max_frames = 1000

for i in range(max_frames):
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # We're done when there are no more frames

    # Save the frame as an image
    image_filename = f'hive2_rpi1_240429-103501Z.jpg{frame_count:04d}.jpg'
    cv2.imwrite(image_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()