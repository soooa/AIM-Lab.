"""
작성자 : 김호진
코드 내용 : bn-wvad 학습 후 plot 및 성능 평가 
"""

import torch
import numpy as np
from dataset_loader_hjk import MicroTubeData
from options import parse_args
import utils
from models import WSAD
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

########################### 오피셜 코드 기반 파라미터 정의 ############################
root_dir = "data\\"
log_path = "log\\"
model_path = "model\\"

lr = [0.0001] * 1000
batch_size = 64
num_workers = 4
num_segments = 200
seed = 2022
model_file = "C:\\Users\\soooa.han\\Desktop\\hjk\\BN-WVAD\\model\\2025-02-27\\tube\\tube_best_2022.pkl" # 모델, 가중치가 저장된 경로
weight_decay = 0.00005
ratio_sample = 0.2
ratio_batch = 0.4

ratios = [16, 32]
kernel_sizes = [1, 1, 1]
len_feature = 1024
num_iters = len(lr)
plot_freq = 5
#########################################################################################

if __name__ == "__main__":

    worker_init_fn = None
    # seed 정의
    utils.set_seed(seed)
    worker_init_fn = np.random.seed(seed)

    # bn-wvad 모델 정의
    net = WSAD(len_feature,flag = "Test", ratio_sample=ratio_sample,
                ratio_batch=ratio_batch,
                ratios=ratios, kernel_sizes=kernel_sizes)
    net = net.cuda() # 모델을 gpu에 올림

    net.load_state_dict(torch.load(model_file)) # 가중치 경로로 접근해서 모델을 로드
    net.eval() #모델을 테스트 모드로 전환


    #성능 평가를 위해 test loader 데이터 사용
    test_loader = data.DataLoader(
        MicroTubeData(root_dir = root_dir, mode = 'Test', num_segments = num_segments, len_feature = len_feature),
            batch_size = 1,
            shuffle = False, num_workers = 0,
            worker_init_fn = worker_init_fn)
    
    load_iter = iter(test_loader)

    label_arr = [] # plot을 위해 label 값을 담은 배열 정의 
    res_arr = [] # plot을 위해 socre 값을 담은 배열 정의 
    test_score = []
    

    max_of_abbormal_data_socre = 12.287409782409668 # 정상중 가장 높은 score기록 한것(segment로 코드 돌릴 시 수정해야 함)
    cnt = 0 # 비정상 개수를 카운트할 변수

    for i in range(len(test_loader.dataset)):

        _data, _label = next(load_iter) # test 데이터 가져옴
        
        _data = _data.cuda() # 가져 온 데이터를 gpu로드
        _label = _label.cuda()# 가져 온 데이터의 레이블을 gpu로드
        res = net(_data) # 데이터 모델 입력

        res_item = res.mean().cpu().item() # 모델에서 나온값을 평균 취한 후 텐서 형태에서 값으로만 가져옴
        label_item = _label.cpu().item() # 레이블값을 텐서 형태에서 값으로만 가져옴

        if label_item == 0: # 정상일때 score값만을 담음
            test_score.append(res_item) 

        if label_item == 1: # 비정상일때 score값만을 담음
            if res_item >= max_of_abbormal_data_socre: # 정상 score중 비정상 score보다 큰 점수가 있으면 이상
                cnt += 1 


        label_arr.append(label_item) #plot을 위한 append
        res_arr.append(res_item) #plot을 위한 append
    
   
    print("비정상 최대 score 값",max(test_score))
    print("오 예측한 정상 데이터", cnt) # 이거 꼭 max_of_abbormal_data_socre 변수 수정한 뒤 확인 해야함

    ### 그래프 생성 ###
    x = range(len(label_arr))
    plt.plot(x, res_arr)
    plt.plot(x, label_arr)
    plt.show()
        
