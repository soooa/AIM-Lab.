import glob
import numpy as np
import os

npy_path = glob.glob("C:\\Users\\soooa.han\\Desktop\\hjk\\BN-WVAD\\stack16\\original_data\\Test\\*.npy") #작은 범위
mp4_path = glob.glob("C:\\Users\\soooa.han\\Desktop\\hjk\\data\\**\\*.mp4", recursive=True) #  큰범위



mp4_list = []

for npy in npy_path:
    new_npy = os.path.splitext(os.path.basename(npy))[0]
    for mp4 in mp4_path:
        new_mp4 = os.path.splitext(os.path.basename(mp4))[0]
        if new_npy.split("_rgb")[0] == new_mp4:
           mp4_list.append(mp4)

print(len(mp4_list))
print(len(npy_path))

