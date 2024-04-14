# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# Updated by WankangZhai (zhai@dlmu.edu.cn)
# ------------------------------------------------------------------------------

# this file we try to use grid method to find the best super-parameters combination


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import datetime

import cupy as cp

import os
import torch
import torch.optim as optim
import prettytable as pt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from cox_nnet import Coxnnet,PartialNLL
from datasets import SurvivalDataset
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger
from sklearn.model_selection import KFold
def train(ini_file,lr_rate,decay_rate,epochs_values,L2_reg):
    ''' Performs training according to .ini file

    :param ini_file: (String) the path of .ini file
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration

    # constructs data loaders based on configuration
    train_dataset = SurvivalDataset(config['train']['h5_file'], is_train=True)
    test_dataset = SurvivalDataset(config['train']['h5_file'], is_train=False)

    train_x = train_dataset.X
    train_y = train_dataset.y
    train_e = train_dataset.e
    test_x = test_dataset.X
    test_y = test_dataset.y
    test_e = test_dataset.e

    all_x = train_x
    all_y = train_y
    all_e = train_e

    # 定义五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ans = []
    # 迭代五次，每次选择不同的训练集和验证集
    for train_index, val_index in kf.split(all_x):
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
            model = Coxnnet(config['network']).to(device)

            criterion = PartialNLL().to(device)
            optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
                model.parameters(), lr=lr_rate,weight_decay=decay_rate)


            # training
            best_c_index = 0
            flag = 0
            train_losses, val_losses, ci_curve = [], [], []

            for epoch in range(1,epochs_values):
                # adjusts learning rate

                # train step

                model.train()
                for X, y, e in train_loader:
                    X = X.to(device).float()
                    y = y.to(device).float()
                    e = e.to(device).float()

                    # makes predictions
                    risk_pred = model(X)
                    train_loss = criterion(risk_pred, y, e)
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
                        valid_loss = criterion(risk_pred, y, e)
                        val_losses.append(valid_loss)
                        valid_c = c_index(-risk_pred, y, e)
                        if best_c_index < valid_c:
                            best_c_index = valid_c
                            ci_curve.append(valid_c)
                            flag = 0
                            # saves the best model
                            torch.save({
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch}, os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))
                        else:
                            flag += 1
                            if flag >= patience:
                                print('a')


                                return best_c_index






        x = happy(train_loader,test_loader)
        ans.append(x)
    print(ans,ini_file)







    return np.average(ans)

if __name__ == '__main__':
    # global settings
    logs_dir = 'logs'
    models_dir = os.path.join(logs_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    logger = create_logger(logs_dir)
    device = torch.device('cuda:0')
    configs_dir = 'configs'
    params = [
        ('TCGA clinical data-KIRC_data', 'TCGA clinical data-KIRC_data.ini'),
        ('TCGA clinical data-LGG_data', 'TCGA clinical data-LGG_data.ini'),
        ('TCGA clinical data-LUAD_data', 'TCGA clinical data-LUAD_data.ini'),
        ('TCGA clinical data-LUSC_data', 'TCGA clinical data-LUSC_data.ini'),
        ('TCGA clinical data-PAAD_data', 'TCGA clinical data-PAAD_data.ini'),
        ('TCGA clinical data-SARC_data', 'TCGA clinical data-SARC_data.ini'),
        ('TCGA clinical data-SKCM_data', 'TCGA clinical data-SKCM_data.ini'),
        ('TCGA clinical data-STAD_data', 'TCGA clinical data-STAD_data.ini'),
        ('TCGA RNA-clinic data-BRCA_data', 'TCGA RNA-clinic data-BRCA_data.ini'),
        ('TCGA RNA-clinic data-CESC_data', 'TCGA RNA-clinic data-CESC_data.ini'),
        ('TCGA RNA-clinic data-COAD_data', 'TCGA RNA-clinic data-COAD_data.ini'),
        ('TCGA RNA-clinic data-HNSC_data', 'TCGA RNA-clinic data-HNSC_data.ini'),
        ('TCGA RNA-clinic data-OV_data', 'TCGA RNA-clinic data-OV_data.ini'),
        ('TCGA RNA-clinic data-UCEC_data', 'TCGA RNA-clinic data-UCEC_data.ini')
            ]

    lr_rate_values = [1e-2,3e-2,1e-3,3e-3,1e-4,3e-4]
    decay_rate_values = [5e-3,1e-3,5e-4,1e-4,5e-5,1e-5]


    patience = 50
    # training
    headers = []
    values = []

    for name, ini_file in params:
        configs_dir = 'configs-coxnnet'
        data = os.path.join(configs_dir, ini_file)
        best_score = 0
        best_model = None  # Initialize the best model

        for lr_rate in lr_rate_values:
            for decay_rate in decay_rate_values:
                for L2_reg in [0]:
                    score = train(os.path.join(configs_dir, ini_file),lr_rate,decay_rate,1000,L2_reg)  # 执行train函数

                    if score > best_score:
                        best_score = score
                        best_params = (lr_rate, decay_rate)
        print("Best Parameters:", best_params)
        print("Best C Index:", best_score)
        with open('coxhappy.txt', 'a') as file:
            file.write(f"Name: {name}\n")
            file.write(f"Best Parameters: {best_params}\n")
            file.write(f"Best C Index: {best_score}\n\n")


    end_time=datetime.datetime.now()

    print("Ended at: "+str(end_time))

