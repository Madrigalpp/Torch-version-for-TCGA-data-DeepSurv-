# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# Updated by WankangZhai (zhai@dlmu.edu.cn)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from cox_nnet import Coxnnet,PartialNLL


import os
import torch
import torch.optim as optim
import prettytable as pt
from torch.multiprocessing import Process
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from networks import DeepSurv
from networks import NegativeLogLikelihood
from datasets import SurvivalDataset
from utils import read_config
from utils import c_index
from networks import *
from utils import adjust_learning_rate
from utils import create_logger
from sklearn.model_selection import KFold
def train(ini_file):
    name = ini_file

    ''' Performs training according to .ini file

    :param ini_file: (String) the path of .ini file
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration

    # constructs data loaders based on configurationy

    train_dataset = SurvivalDataset(config['train']['h5_file'], is_train=True)
    test_dataset = SurvivalDataset(config['train']['h5_file'], is_train=False)

    train_x = train_dataset.X
    train_y = train_dataset.y
    train_e = train_dataset.e
    test_x = test_dataset.X
    test_y = test_dataset.y
    test_e = test_dataset.e

    all_x = np.concatenate((train_x, test_x), axis=0)
    all_y = np.concatenate((train_y, test_y), axis=0)
    all_e = np.concatenate((train_e, test_e), axis=0)

    # 定义五折交叉验证

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ans = []
    ans_pred = []
    # 迭代五次，每次选择不同的训练集和验证集
    i = 0
    for train_index, val_index in kf.split(all_x):
        i+= 1
        train_fold_x, val_fold_x = all_x[train_index], all_x[val_index]
        train_fold_y, val_fold_y = all_y[train_index], all_y[val_index]
        train_fold_e, val_fold_e = all_e[train_index], all_e[val_index]
        train_dataset = TensorDataset(torch.Tensor(train_fold_x), torch.Tensor(train_fold_y),
                                      torch.Tensor(train_fold_e))
        test_dataset = TensorDataset(torch.Tensor(val_fold_x), torch.Tensor(val_fold_y),
                                        torch.Tensor(val_fold_e))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_dataset.__len__())
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_dataset.__len__())

        def happy(train_loader,test_loader):
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = DeepSurv(config['network']).to(device)

            criterion = ZLs(config['network']).to(device)
            optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
                model.parameters(), lr=config['train']['learning_rate'])


            # training
            best_c_index = 0
            flag = 0
            train_losses, val_losses, ci_curve = [], [], []

            for epoch in range(1, config['train']['epochs']+1):
                # adjusts learning rate

                # train step

                model.train()
                for X, y, e in train_loader:
                    X = X.to(device).float()
                    y = y.to(device).float()
                    e = e.to(device).float()

                    # makes predictions
                    risk_pred = model(X)
                    train_loss = criterion(risk_pred, y, e, model)
                    train_losses.append(train_loss)
                    train_c = c_index(-risk_pred, y, e)
                    # updates parameters
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                # valid step
                model.eval()
                for X, y, e in test_loader:
                    X = X.to(device).float()
                    y = y.to(device).float()
                    e = e.to(device).float()

                    # makes predictions
                    with torch.no_grad():
                        risk_pred = model(X)
                        valid_loss = criterion(risk_pred, y, e, model)
                        val_losses.append(valid_loss)
                        valid_c = c_index(-risk_pred, y, e)
                        if best_c_index < valid_c:
                            best_c_index = valid_c
                            ci_curve.append(valid_c)
                            flag = 0

                        else:
                            flag += 1
                            if flag >= 10:
                                print('Early stopping!')


                                return best_c_index,train_losses,val_losses,ci_curve,risk_pred,y,e





                # notes that, train loader and valid loader both have one batch!!!

        x,y,z,k,risk_pred,y,e = happy(train_loader,test_loader)
        risk_pred = risk_pred.cpu().numpy()
        y = y.cpu().numpy()
        e = e.cpu().numpy()
        result = np.concatenate((risk_pred, y, e), axis=1)

        df = pd.DataFrame(result, columns=['risk_pred', 'y', 'e'])
        ans_pred.append(df)

        folder_name = "ults"
        os.makedirs(folder_name, exist_ok=True)

        # 拼接文件路径
        file_path = os.path.join(folder_name, f"ans{name},{i}.csv")

        # 将数据框保存为CSV文件
        df.to_csv(file_path, index=False)

        print(f"Data saved to: {file_path}")

        ans.append(x)

    return np.average(ans),ans,ans_pred

def train_process(config_file):
    best_c_index, ans,ans_pred = train(config_file)
    name = os.path.splitext(os.path.basename(config_file))[0].split('-')[-1]
    tumor_type = os.path.splitext(os.path.basename(config_file))[0].split('-')[-1]

    for i, df in enumerate(ans_pred):
        output_path = f'output_file_{i + 1}.csv'
        df.to_csv(output_path, index=False)


    with open('coxnew.txt', 'a') as f:
        print("The best valid c-index: {}".format(tumor_type), file=f)

        print("The best valid c-index: {}".format(best_c_index), file=f)

        print("The best valid c-index: {}".format(ans), file=f)




        print("\n", file=f)


if __name__ == '__main__':

    import time

    start_time = time.time()



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # configs_dir = 'configs - 副本'
    configs_dir = 'configs-coxnnet'

    params = [
        'TCGA clinical data-LIHC_data.ini','TCGA clinical data-KIRC_data.ini', 'TCGA clinical data-LGG_data.ini',
              'TCGA clinical data-LUAD_data.ini',
        'TCGA clinical data-LUSC_data.ini',
              'TCGA clinical data-PAAD_data.ini', 'TCGA clinical data-SARC_data.ini',
              'TCGA clinical data-SKCM_data.ini', 'TCGA clinical data-STAD_data.ini',
              'TCGA RNA-clinic data-BRCA_data.ini', 'TCGA RNA-clinic data-CESC_data.ini',
              'TCGA RNA-clinic data-COAD_data.ini', 'TCGA RNA-clinic data-HNSC_data.ini',
              'TCGA RNA-clinic data-OV_data.ini', 'TCGA RNA-clinic data-UCEC_data.ini'
              ]

    patience = 30
    processes = []

    for param in params:
        process = Process(target=train_process, args=(os.path.join(configs_dir, param),))
        processes.append(process)
        process.start()

    for process in processes:

        process.join()








