"""
작성자 : 김호진
코드 내용 : bn-wvad 를 위한 데이터 가져오는 부분 
"""


import torch
import torch.utils.data as data
import os
import numpy as np
import utils
import glob


class MicroTubeData(data.Dataset):
    def __init__(self, root_dir, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)

        self.data_path = root_dir # "data" 폴더(data 폴더 생성후 Train/ Test 폴더 생성 후 .npy 데이터 넣어주면 됨)
        self.mode = mode          # 모델이 train인지 test 인지 지정
        self.num_segments = num_segments # 논문 참조
        self.len_feature = len_feature # 논문 참조
        self.feature_path = self.data_path
        split_path = os.path.join(self.feature_path, mode) # "Train 시, data/Train 접근/ Test시 data/Test 접근"
        self.vid_list = []

        # 모든 npy 파일 다 가져오기
        #한 폴더 내부 모든 폴더를 순회하며(recursive = True), .npy라는 확장자를 모두 가져옴 
        self.vid_list = glob.glob(os.path.join(split_path, "*.npy"), recursive=True) 
        self.mp4_list = glob.glob("C:\\Users\\soooa.han\\Desktop\\hjk\\data\\**\\*.mp4", recursive=True)

        #train시, npy 파일중 정상 데이터와 비정상 데이터 구분(normal_dataloader, abnormal_dataloader로 분할 해야하기 때문)
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = [vid for vid in self.vid_list if "normal" in vid]
            elif is_normal is False:
                self.vid_list = [vid for vid in self.vid_list if "normal" not in vid]


    def __len__(self):
        return len(self.vid_list) # 잡아온 데이터 전체 길이 return
    
    def __getitem__(self, index):
        data, label, vid_name, mp4_name = self.get_data(index) # 실질 적으로 데이터를 뱉어내는 함수
        if self.mode == 'Test':
            return data, label, vid_name, mp4_name
        
        elif self.mode == 'Train':
            return data, label
        
    def get_data(self, index):
        vid_name = self.vid_list[index] #self.vid_list는 전체 npy 경로를 담은 배열
        label = 0 # 디폴트 레이블 0(npy 수준의 레이블)
        
        mp4_return = []

        for mp4 in self.mp4_list:
            mp4_path = os.path.splitext(os.path.basename(mp4))[0]
            if os.path.splitext(os.path.basename(vid_name))[0].split("_rgb")[0] == mp4_path:
                mp4_return.append(mp4)


        if "normal" not in vid_name: # 파일 이름중 normal이 미포함 되어 있으면 label을 1로 지정
            label = 1
        
        video_feature = np.load(vid_name).astype(np.float32) #npy 파일을 열어주는 메소드

        # num_segment, feature_len ---> 논문 참조
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)
            sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=np.uint16) #한 i3d 피쳐 분할

            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]: 
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0) # * (num_segment / feature_len) 만큼 길이 조절(평균연산을 이용)
                    
            video_feature = new_feature
        return video_feature, label, vid_name, mp4_return







