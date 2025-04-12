import os
import glob
import cv2
import numpy as np


path = glob.glob("./stack16/original_data/Test/*.npy")

for i in path:
    data = np.load(i)
    print(data.shape)

#(8, 1024) / 4초 / 30 fps ->  총 120 fps -? 120 /16  = 7.5
# 한 영상 당 총 8개의 스니펫
# 8장의 이미지 당 하나의 score


video_path = "C:/Users/soooa.han/Desktop/hjk/data/highspeed/blue_highspeed2_35mpm/blue_highspeed2_35mpm_segment1.mp4"
cap = cv2.VideoCapture(video_path)

#150개
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap_cnt = 0
height = 256
width = 256

# cnt = 0
# for i in range(2, 150):
#     if i % 16 == 0 or i % 16 == 1:
#         print(i)

#     else:
#         print("############")

show_cnt = 0
while cap.isOpened:
    cap_cnt += 1
    ret, frame = cap.read()
    if not ret:
        break


    if (cap_cnt % 16 == 0 or cap_cnt % 16 == 1) and cap_cnt > 3:
        show_cnt += 1
        frame = cv2.resize(frame, (width, height))
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(show_cnt)