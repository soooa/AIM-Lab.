"""
작성자 : 김호진
코드 내용 : bn-wvad main문
"""



import numpy as np
import torch.utils.data as data
import utils
import time


from train import train
from losses import LossComputer
from test_hjk import test
from models import WSAD

from dataset_loader_hjk import *
from tqdm import tqdm

########### hyper parameter(파라미터 오피셜 코드 가져옴) ##########
root_dir = "data/"
log_path = "log/"
model_path = "model/"

lr = [0.0001] * 1000
batch_size = 64
num_workers = 4
num_segments = 200
seed = 2022

weight_decay = 0.00005
ratio_sample = 0.2
ratio_batch = 0.4

ratios = [16, 32]
kernel_sizes = [1, 1, 1]
len_feature = 1024
num_iters = len(lr)
plot_freq = 5
#################################################

localtime = time.localtime()
time_ymd = time.strftime("%Y-%m-%d", localtime)
time_hms = time.strftime("%H:%M:%S", localtime)

if __name__ == "__main__":
    log_path = os.path.join(log_path, time_ymd, 'tube') # train log가 저장될 폴더 신경x
    model_path = os.path.join(model_path, time_ymd, 'tube') # 모델 저장될 폴더

    # 위에서 지정 한 경로가 없을 시, 새로 폴더 만들어줌
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # seed 생성 딱히 신경 안써도됨 random한 난수 생성 정도?
    worker_init_fn = None
    if seed >= 0:
        utils.set_seed(seed)
        worker_init_fn = np.random.seed(seed)


    # bn-wvad 모델 불러오는 부분 
    net = WSAD(len_feature,flag = "Test", ratio_sample=ratio_sample,
                ratio_batch=ratio_batch,
                ratios=ratios, kernel_sizes=kernel_sizes)
    #모델을 gpu에 올림
    net = net.cuda()

     # train npy 중 정상 npy만 불러오는 부분
    normal_train_loader = data.DataLoader(
        MicroTubeData(root_dir = root_dir, mode = 'Train', num_segments = num_segments, len_feature = len_feature, is_normal = True),
            batch_size = batch_size,
            shuffle = True, num_workers = num_workers,
            worker_init_fn = worker_init_fn)

    # train npy 중 정상 npy만 불러오는 부분
    abnormal_train_loader = data.DataLoader( 
        MicroTubeData(root_dir = root_dir, mode='Train', num_segments = num_segments, len_feature = len_feature, is_normal = False),
            batch_size = batch_size,
            shuffle = True, num_workers = num_workers,
            worker_init_fn = worker_init_fn)

    # test npy 중 정상 npy만 불러오는 부분
    test_loader = data.DataLoader( 
        MicroTubeData(root_dir = root_dir, mode = 'Test', num_segments = num_segments, len_feature = len_feature),
            batch_size = 1,
            shuffle = False, num_workers = num_workers,
            worker_init_fn = worker_init_fn)

    # loss 함수 정의
    criterion = LossComputer()

    # optimizer 정의(파라미터 오피셜 코드 가져옴)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr[0],
        betas = (0.9, 0.999), weight_decay = weight_decay)

    # 처음 score 를 무한대로 지정해 줌
    init_score = 9999999

    for step in tqdm(range(1, num_iters + 1), total = num_iters, dynamic_ncols = True ): #train 돌리면 로딩하는거 마냥 돌아가는데 그런 기능 해주는 라이브러리(tqdm)


######################### 오피셜 코드 가져옴 ###########################
#     train 시, normal과 abnormal을 동시에 잡아오는데, 데이터 불균형이 있을 시 오류를 방지하기 위함
        if step > 1 and lr[step - 1] != lr[step - 2]: # lr 잡아옴 -> 전부 0.0001임
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr[step - 1]

        if (step - 1) % len(normal_train_loader) == 0:  # normal data 잡아오는 부분
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0: # abnormal 잡아오는 부분
            abnormal_loader_iter = iter(abnormal_train_loader)
########################################################################

        losses = train(net, normal_loader_iter,abnormal_loader_iter, optimizer, criterion) #loss 함수 지정
        
        if step % 10 == 0: # 10 epoch 당 test 데이터에서 배출되는 score값을 조사한 후, 성능이 개선되면 가중치를 저장
            total_loss = test(net, test_loader)
            print('test_total_cost is ', total_loss)

            if init_score > total_loss:
                init_score = total_loss

                torch.save(net.state_dict(), os.path.join(model_path, "tube_best_{}.pkl".format(seed)))
                print('saved!')()
