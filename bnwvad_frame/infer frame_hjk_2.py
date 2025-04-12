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
import cv2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

########################### 오피셜 코드 기반 파라미터 정의 ############################
root_dir = "stack16\\original_data"
log_path = "log\\"
model_path = "model\\"

lr = [0.0001] * 1000
batch_size = 1
num_workers = 4
num_segments = 200
seed = 2022
model_file = "C:\\Users\\soooa.han\\Desktop\hjk\\BN-WVAD\\stack16_original.pkl" # 모델, 가중치가 저장된 경로
weight_decay = 0.00005
ratio_sample = 0.2
ratio_batch = 0.4

ratios = [16, 32]
kernel_sizes = [1, 1, 1]
len_feature = 1024
num_iters = len(lr)
plot_freq = 5
class_map = {   # 클래스 이름, 클래스 개수, 클래스의 anomaly socre, anomaly_score의 평균
    "blue_highspeed1_38mpm" : [0,0,0],
    "blue_highspeed2_35mpm" : [0,0,0],
    "blue_lowspeed_5.5mpm" : [0,0,0],
    "blue_normal" : [0,0,0],

    "opacity_40mpm" : [0,0,0],
    "opacity_normal" : [0,0,0],
    "opacity_temp_p20_40mpm" : [0,0,0],

    "transparency_40mpm" : [0,0,0],
    "transparency_normal" : [0,0,0],
    "transparency_temp_p30_40mpm" : [0,0,0],
    "transperant_highspeed_35mpm" : [0,0,0],
    "transperant_lowspeed_5.5mpm" : [0,0,0],
    "transperant_normal" : [0,0,0]
    }
#########################################################################################

def get_sub_metrics(frame_predict, frame_gt):
    anomaly_mask = frame_gt
    sub_predict = frame_predict[anomaly_mask]
    sub_gt = frame_gt[anomaly_mask]
    
    fpr,tpr,_ = roc_curve(sub_gt, sub_predict)
    auc_sub = auc(fpr, tpr)

    precision, recall, th = precision_recall_curve(sub_gt, sub_predict)
    ap_sub = auc(recall, precision)
    return auc_sub, ap_sub


def get_metrics(frame_predict, frame_gt):
    metrics = {}
    
    fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
    metrics['AUC'] = auc(fpr, tpr)
    
    precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
    metrics['AP'] = auc(recall, precision)

    auc_sub, ap_sub = get_sub_metrics(frame_predict, frame_gt)
    metrics['AUC_sub'] = auc_sub
    metrics['AP_sub'] = ap_sub

    return metrics

################### GPT ##############################
def plot_to_image(y_data, height, cnt):
    fig, ax = plt.subplots(figsize=(3, 3))
    canvas = FigureCanvas(fig)

    ax.plot(y_data, color='red')

    ax.tick_params(axis='both', labelsize=8)

    canvas.draw()

    width, height_ = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(int(height_), int(width), 3)
    plt.close(fig)

    scale = height / img.shape[0]
    new_w = int(img.shape[1] * scale)
    img = cv2.resize(img, (new_w, height))
    return img
#####################################################


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


    # 성능 평가를 위해 test loader 데이터 사용
    test_loader = data.DataLoader(
        MicroTubeData(root_dir = root_dir, mode = 'Test', num_segments = num_segments, len_feature = len_feature),
            batch_size = 1,
            shuffle = True, num_workers = 0,
            worker_init_fn = worker_init_fn)
    
    load_iter = iter(test_loader)

    label_arr = [] # plot을 위해 label 값을 담은 배열 정의 
    res_arr = [] # plot을 위해 socre 값을 담은 배열 정의 
    test_score = []
    
    max_of_abbormal_data_socre = 12.287409782409668 # 정상중 가장 높은 score기록 한것(segment로 코드 돌릴 시 수정해야 함)
    cnt = 0 # 비정상 개수를 카운트할 변수
    frame_predict = []
    frame_predict_label = []
    mp4_path_list = []


    for i in range(len(test_loader.dataset)):

        _data, _label, vid_name, mp4_path = next(load_iter) # test 데이터 가져옴
        
        mp4_path_list.append(mp4_path[0][0])

        _data = _data.cuda() # 가져 온 데이터를 gpu로드
        _label = _label.cuda()# 가져 온 데이터의 레이블을 gpu로드
        res = net(_data) # 데이터 모델 입력

        res_item = res.detach().cpu().numpy() # 모델에서 나온값을 평균 취한 후 텐서 형태에서 값으로만 가져옴
        label_item = _label.detach().cpu().numpy() # 레이블값을 텐서 형태에서 값으로만 가져옴          

        fpre_ = np.repeat(res_item, 16)
        fpre_label_ = np.repeat(label_item, 128)

        frame_predict.append(fpre_)
        frame_predict_label.append(fpre_label_)

        label_arr.append(label_item) #plot을 위한 append
        res_arr.append(res_item) #plot을 위한 append


    cap0 = cv2.VideoCapture(mp4_path_list[0])
    width  = 512
    height = 512
    fps    = cap0.get(cv2.CAP_PROP_FPS)
    cap0.release()
    
    
    y_values = []
    score_idx = 0
    score_idx2 = 0
    cnt = 0
    for video_path in mp4_path_list:
        y_values = []
        score_idx2 = 0
        cap = cv2.VideoCapture(video_path)
        while score_idx2 < 127:
            cnt += 1
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (512, 512))
            
            score = frame_predict[score_idx][score_idx2]
            y_values.append(score)

            graph_img = plot_to_image(y_values, height, cnt)

            combined = np.hstack((frame, graph_img))
            cv2.imshow("Video + Graph", combined)
            if score_idx2 % 16 == 0 or score_idx2 == 1:
                cv2.imwrite(f'fig/{score_idx}_{score_idx2}.png', combined) 
            score_idx2 += 1 

            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
            print(score_idx, score_idx2, frame_predict[score_idx][score_idx2])
        score_idx += 1 
        cap.release()
        

    cv2.destroyAllWindows()
    


    

    


            
