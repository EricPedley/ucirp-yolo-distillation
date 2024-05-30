import cv2 as cv
import os
from tqdm import tqdm

video_filenames = os.listdir("videos")
os.makedirs("frames", exist_ok=True)

for video_filename in video_filenames:
    video = cv.VideoCapture("videos/" + video_filename)
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count)):
        success, frame = video.read()
        if i%10 != 0:
            continue
        if not success:
            break
        cv.imwrite("frames/" + video_filename + "_" + str(i) + ".png", frame)
    video.release()