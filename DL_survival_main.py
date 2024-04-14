# ------------------------------------------------------------------------------
# --coding='utf-8'--
# cited czifan (czifan@pku.edu.cn)  version
# if you have any problem， please contact zhai（wzhai2@uh.edu）
# ------------------------------------------------------------------------------

# here is a main function for deep learning survival analysis.
# we combine 5 fold cross-validation and draw the train/test curve.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import cupy as cp

import os
import torch
import torch.optim as optim
import prettytable as pt
import warnings
from DAE降低维度 import DenoisingAutoencoder
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from networks import DeepSurv
from networks import NegativeLogLikelihood
from networks import *
from datasets import SurvivalDataset
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger
from sklearn.model_selection import KFold
from resnet_network import *
def train(ini_file):

    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration
    model = DeepSurv(config['network']).to(device)
    criterion = ZLs(config['network']).to(device)

    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])
    # constructs data loaders based on configuration
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
    # all_x = aaa

    # 定义五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

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





        # training
        best_c_index = 0
        flag = 0
        train_losses, val_losses, ci_curve = [], [], []


        for epoch in range(1, config['train']['epochs']+1):
            # adjusts learning rate
            lr = adjust_learning_rate(optimizer, epoch,
                                      config['train']['learning_rate'],
                                      config['train']['lr_decay_rate'])
            # train step
            model.train()
            for X, y, e in train_loader:
                X = X.to(device).float()
                y = y.to(device).float()
                e = e.to(device).float()
                risk_pred = model(X)

                # makes predictions
                train_loss = criterion(risk_pred, y, e,model)
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


                    valid_loss = criterion(risk_pred, y, e,model)

                    val_losses.append(valid_loss)
                    valid_c = c_index(-risk_pred, y, e)
                    ci_curve.append(valid_c)

                    if best_c_index < valid_c:
                        best_c_index = valid_c
                        flag = 0
                        # saves the best model
                    else:
                        flag += 1

                        if flag >= patience:
                            print('Early stopping!')

                            return best_c_index, train_losses, val_losses, ci_curve
            # notes that, train loader and valid loader both have one batch!!!
            print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(
                epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)



        return best_c_index

if __name__ == '__main__':
    # global settings
    logs_dir = 'logs'
    models_dir = os.path.join(logs_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    logger = create_logger(logs_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    configs_dir = 'configs-coxnnet'
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
        ('TCGA RNA-clinic data-UCEC_data', 'TCGA RNA-clinic data-UCEC_data.ini'),
        ('TCGA clinical data-LIHC_data', 'TCGA clinical data-LIHC_data.ini'),
    ]
    patience = 300
    # training
    headers = []
    values = []
    for name, ini_file in params:
        logger.info('Running {}({})...'.format(name, ini_file))
        best_c_index,train_losses,val_losses,ci_curve = train(os.path.join(configs_dir, ini_file))
        fig, axs = plt.subplots(3, figsize=[15, 20])
        # Assuming train_losses is a list of PyTorch tensors
        train_losses = cp.asnumpy(torch.Tensor(train_losses).cpu())
        val_losses = cp.asnumpy(torch.Tensor(val_losses).cpu())
        folder_path = '作图数据'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)



        np.save(os.path.join(folder_path, 'brca_Cox_TR.npy'), train_losses)
        np.save(os.path.join(folder_path, 'brca_Cox_TST.npy'), val_losses)
        np.save(os.path.join(folder_path, 'brca_Cox_CIN.npy'), ci_curve)



        axs[0].plot(train_losses), axs[0].set_title("Average Train Loss{}".format(name))
        axs[1].plot(val_losses), axs[1].set_title("Average Validation Loss{}".format(name))
        axs[2].plot(ci_curve), axs[2].set_title("Concordance Index{}".format(name))
        fig.savefig(
            "img/valid_{}_lr={}.png".format(ini_file, name ))
        plt.show()
        headers.append(name)
        values.append('{:.6f}'.format(best_c_index))
        print('')
        logger.info("The best valid c-index: {}".format(best_c_index))
        logger.info('')
    # prints results
    tb = pt.PrettyTable()
    tb.field_names = headers
    tb.add_row(values)
    logger.info(tb)

