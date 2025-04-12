"""
작성자 : 김호진
코드 내용 : bn-wvad 학습 중 iter 마다 성능 평가를 위한 코드
"""



from options import *
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings

warnings.filterwarnings("ignore")


def test(test_loader, net):
    net.eval()
    net.flag = "Test" #flag : test
    load_iter = iter(test_loader)
    normal_cost_arr = []  # normal_cost를 담을 배열
    abnormal_cost_arr = [] # abnormal_cost를 담을 배열
    
    for i in range(len(test_loader.dataset)):
        _data, _label, _, _ = next(load_iter) # 데이터를 순차적으로 불러옴
        
        ##gpu에 올리고 데이터 모델에 넣음
        _data = _data.cuda()
        _label = _label.cuda()
        res = net(_data)
        res = res.mean().item() # 모델에서 배출된값 평균 취하고 텐서에서 스칼라 값(float)으로 변환

        if _label.cpu().item() == 0: #정상인 경우 담을 배열
            normal_cost_arr.append(res) 

        elif _label.cpu().item() == 1: # 비정상인 경우 담을 배열
            abnormal_cost_arr.append(res)

    total_normal_cost = sum(normal_cost_arr)/len(normal_cost_arr) # 전체 정상 score를 구해서 평균
    total_abnormal_cost = sum(abnormal_cost_arr)/len(abnormal_cost_arr) # 전체 비저상 score를 구해서 평균

    total_cost = total_normal_cost / total_abnormal_cost # 전체 정상 평균 / 전체 비정상 평균(total score가 낮을수록 좋은 모델이라고 판단하려는 의도로 작성)

    
    return total_cost
