# ------------------------------------------------------------------------------
# --coding='utf-8'--
# cited czifan (czifan@pku.edu.cn)  version
# if you have any problem， please contact zhai（wzhai2@uh.edu）
# ------------------------------------------------------------------------------

# here is a network  for deep learning survival analysis.
# we realize deepsurv and coxnnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.nn import init
import numpy as np
import torch
import torch.nn as nn
import math
class Happy_Loss(nn.Module):
    def __init__(self, config):
        super(Happy_Loss, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        def R_set(x):
            '''Create an indicator matrix of risk sets, where T_j >= T_i.
            Note that the input data have been sorted in descending order.
            Input:
                x: a PyTorch tensor that the number of rows is equal to the number of samples.
            Output:
                indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
            '''
            n_sample = x.size(0)
            matrix_ones = torch.ones(n_sample, n_sample)
            indicator_matrix = torch.tril(matrix_ones)

            return (indicator_matrix)

        def neg_par_log_likelihood(pred, ytime, yevent):
            '''Calculate the average Cox negative partial log-likelihood.
            Note that this function requires the input data have been sorted in descending order.
            Input:
                pred: linear predictors from trained model.
                ytime: true survival time from load_data().
                yevent: true censoring status from load_data().
            Output:
                cost: the cost that is to be minimized.
            '''
            n_observed = yevent.sum(0)
            ytime_indicator = R_set(ytime)
            ###if gpu is being used
            if torch.cuda.is_available():
                ytime_indicator = ytime_indicator.cuda()
            ###
            risk_set_sum = ytime_indicator.mm(torch.exp(pred))
            diff = pred - torch.log(risk_set_sum)
            sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
            cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))

            return (cost)
        # risk_pred = torch.mean(risk_pred, dim=1, keepdim=True)

        l2_loss = self.reg(model)
        cox_ph_loss = neg_par_log_likelihood(risk_pred, y, e)


        return cox_ph_loss



class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class PartialNLL(torch.nn.Module):
    def __init__(self):
        super(PartialNLL, self).__init__()

    # Use matrix to avoid nested loops, for faster calculation
    def _make_R_rev(self, y):
        device = y.device
        R = np.zeros((y.shape[0], y.shape[0]))
        y = y.cpu().detach().numpy()
        for i in range(y.shape[0]):
            idx = np.where(y >= y[i])
            R[i, idx] = 1
        return torch.tensor(R, device=device)

    # 按行来查找， 只要比自己大的都是1， 但是为什么要这么做呢
    # 如这一行： 首先我们定义了 一个随机的矩阵
    #对于第一行 是3 3>1,  3>2, 所以第一行的第一列，第三列，第四列都是0


    # tensor([3., 1., 4., 2.], requires_grad=True)
    #
    # Output
    #
    # tensor([[1., 0., 1., 0.],
    #         [1., 1., 1., 1.],
    #         [0., 0., 1., 0.],
    #         [1., 0., 1., 1.]], dtype=torch.float64)




    def forward(self, theta, time, observed):
        R_rev = self._make_R_rev(time).to(theta.device)

        # 首先对所有的 t 找到他的R_rev 矩阵

        # 所有比自己小的都是0， 比自己大的都是1

        # 因此 生存时间比自己少的 就可以忽略不管了

        # 成功实现 ti > tj 的情况


        exp_theta = torch.exp(theta)
        num_observed = torch.sum(observed)
        loss = -torch.sum((theta.reshape(-1) - torch.log(torch.sum((exp_theta * R_rev.t()), 0))) * observed) / num_observed
        if np.isnan(loss.data.tolist()):
            for a, b in zip(theta, exp_theta):
                print(a, b)
        return loss # 返回的是交叉熵的损失




class Reduction_Loss(nn.Module):
    def __init__(self, config):
        super(Reduction_Loss, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        risk_pred = torch.mean(risk_pred, dim=1, keepdim=True)
        def cox_ph_loss(log_h, durations, events, eps: float = 1e-5):
            durations = torch.tensor(durations)

            values, idx = torch.sort(durations, descending=True)
            events = events[idx]
            log_h = log_h[idx]
            return cox_ph_loss_sorted(log_h, events, eps)

        def cox_ph_loss_sorted(log_h, events, eps: float = 1e-7):
            if events.dtype is torch.bool:
                events = events.float()
            events = torch.tensor(events)
            log_h = torch.tensor(log_h)
            eps = torch.tensor(eps)
            events = events.view(-1)
            log_h = log_h.view(-1)
            gamma = log_h.max()
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma).to(device)
            events = events.to(device)
            return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


        l2_loss = self.reg(model)
        cox_ph_loss = cox_ph_loss(risk_pred, y, e)


        return cox_ph_loss + l2_loss


def R_set(x):
            '''Create an indicator matrix of risk sets, where T_j >= T_i.
            Note that the input data have been sorted in descending order.
            Input:
                x: a PyTorch tensor that the number of rows is equal to the number of samples.
            Output:
                indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
            '''
            n_sample = x.size(0)
            matrix_ones = torch.ones(n_sample, n_sample)
            indicator_matrix = torch.tril(matrix_ones)

            return (indicator_matrix)


def c_indexpp(pred, ytime, yevent):
    '''Calculate concordance index to evaluate models.
    Input:
        pred: linear predictors from trained model.
        ytime: true survival time from load_data().
        yevent: true censoring status from load_data().
    Output:
        concordance_index: c-index (between 0 and 1).
    '''
    n_sample = len(ytime)
    ytime_indicator = R_set(ytime)
    ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
    ###T_i is uncensored
    censor_idx = (yevent == 0).nonzero()
    zeros = torch.zeros(n_sample)
    ytime_matrix[censor_idx, :] = zeros
    ###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
    pred_matrix = torch.zeros_like(ytime_matrix)
    for j in range(n_sample):
        for i in range(n_sample):
            if pred[i] < pred[j]:
                pred_matrix[j, i] = 1
            elif pred[i] == pred[j]:
                pred_matrix[j, i] = 0.5

    concord_matrix = pred_matrix.mul(ytime_matrix)
    ###numerator
    concord = torch.sum(concord_matrix)
    ###denominator
    epsilon = torch.sum(ytime_matrix)
    ###c-index = numerator/denominator
    concordance_index = torch.div(concord, epsilon)
    ###if gpu is being used
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
    ###
    return (concordance_index)



class CoxNNetLoss(torch.nn.Module):
    def __init__(self,config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, hazard_pred, durations, events,model):
        def make_R_rev(y):
            device = y.device
            R = np.zeros((y.shape[0], y.shape[0]))
            y = y.cpu().detach().numpy()
            for i in range(y.shape[0]):
                idx = np.where(y >= y[i])
                R[i, idx] = 1
            return torch.tensor(R, device=device)


        # durations: Tensor
        # events: Tensor
        # # durations, events = y.T
        # durations, events = y

        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(durations)
        R_mat = make_R_rev(durations).to(durations.device)


        R_mat = torch.as_tensor(R_mat)#, device=hazard_pred.device)


        theta = hazard_pred.reshape(-1)
        exp_theta= torch.exp(theta)
        loss_cox = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * events,
        )
        l2_loss = self.reg(model)

        return loss_cox+l2_loss #.to( device=hazard_pred.device)

class AECox(nn.Module):
    def __init__(self,config):
        super(AECox, self).__init__()
        # Check number of features!
        # torch.load(" <file_path> ")["model_state_dict"])
        self.num_features = config['dims'][0]
        self.dropout_rate=0.3
        self.hidden_nodes=128

        # self.ae = torch_maven.AE(num_features=num_features, gcn_mode=gcn_mode)
        # self.ae = torch_maven.AE(config, logger, num_features, gcn_mode)


        self.encode = nn.Sequential(
            nn.Linear(self.num_features, 4096),
            nn.ReLU(),

            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, 128),
            nn.Linear(128, 1)
            # nn.Linear(128,4096),
            # nn.Dropout(self.dropout_rate),
            # nn.ReLU(),
            # nn.Linear(4096,self.num_features)
        )

    def forward(self, x):
        x = self.encode(x)

        return x


class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if self.norm: # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            layers.append(eval('nn.{}()'.format(self.activation)))
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)
class AEcox(nn.Module):
    def __init__(self, config):
        super(AEcox, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        input_size = config['dims'][0]
        # builds network
        self.encode = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, 1024))
        self.decoder = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(4096, input_size))
        self.cox = nn.Sequential(
            nn.Linear(input_size,128),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128,1))

    def forward(self, x):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        x = self.encode(x)
        x = self.cox(x)
        return x
    def aemodel(self,x):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        x = self.encode(x)
        x = torch.relu(x)
        x = self.decoder(x)
        return x


class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        # risk_pred = torch.mean(risk_pred, dim=1, keepdim=True)
        def cox_ph_loss(log_h, durations, events, eps: float = 1e-7):
            durations = torch.tensor(durations)

            values, idx = torch.sort(durations, descending=True)
            events = events[idx]
            log_h = log_h[idx]
            return cox_ph_loss_sorted(log_h, events, eps)

        def cox_ph_loss_sorted(log_h, events, eps: float = 1e-7):
            if events.dtype is torch.bool:
                events = events.float()
            events = torch.tensor(events)
            log_h = torch.tensor(log_h)
            eps = torch.tensor(eps)
            events = events.view(-1)
            log_h = log_h.view(-1)
            gamma = log_h.max()
            log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
            return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())




        mask = torch.ones(y.shape[0], y.shape[0])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        mask[(y.T - y) > 0] = 0
        mask = mask.to(device)
        risk_pred = risk_pred.to(device)
        log_loss = torch.exp(risk_pred) * mask
        # log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.sum(log_loss, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        cox_ph_loss = cox_ph_loss(risk_pred, y, e)


        return cox_ph_loss + l2_loss


class ZLs(nn.Module):
    def __init__(self, config):
        super(ZLs, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):


        mask = torch.ones(y.shape[0], y.shape[0])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        mask[(y.T - y) > 0] = 0
        mask = mask.to(device)
        # risk_pred = torch.mean(risk_pred, dim=1, keepdim=True)
        # print(risk_pred)

        # risk_pred = risk_pred.reshape(-1)


        log_loss = torch.exp(risk_pred) * mask
        # log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.sum(log_loss, dim=0)
        log_loss = torch.log(log_loss ).reshape(-1, 1)
        log_loss = log_loss.to(device)
        risk_pred = risk_pred.to(device)
        e = torch.tensor(e)
        e = e.to(device)


        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)




        return neg_log_loss +l2_loss



class ZLstime(nn.Module):
    def __init__(self, config):
        super(ZLstime, self).__init__()
        self.L2_reg = config['l2_reg']


        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        time_range = y.max() - y.min()
        average_interval = time_range / 5
        indices = torch.nonzero(y < 3*average_interval).squeeze()
        risk_pred_low = risk_pred
        risk_pred_low[indices] = 0
        log_loss = torch.exp(risk_pred_low)


        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



        # log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.sum(log_loss, dim=0)
        # 这里有问题
        log_loss = torch.log(log_loss).reshape(-1, 1)
        log_loss = log_loss.to(device)
        risk_pred = risk_pred.to(device)
        e = torch.tensor(e)
        e = e.to(device)



        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)

        combine = torch.cat((risk_pred, log_loss), 1)
        print(combine)
        l2_loss = self.reg(model)




        return neg_log_loss +l2_loss


class ZLsv2(nn.Module):
    def __init__(self, config):
        super(ZLsv2, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        min_y = torch.min(y)
        max_y = torch.max(y)
        scaled_y = 1 + (y - min_y) * (10) / (max_y - min_y)
        import torch.distributions as dist
        import matplotlib.pyplot as plt
        shape_parameter = 2.0
        scale_parameter = 1.0
        weibull_dist = dist.Weibull(shape_parameter, scale_parameter)
        # print(y)
        print(min_y)
        pdf_values = weibull_dist.log_prob(y).exp()


        # 归一化
        normalized_tensor = scaled_y
        normalized_tensor = pdf_values.reshape(-1, 1)

        mask = torch.ones(y.shape[0], y.shape[0])
        maskb = torch.ones(y.shape[0], y.shape[0])


        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        maskb = maskb.to(device)
        mask[(y.T - y) > 0] = 0
        mask = mask.to(device)

        log_lossa = torch.exp(risk_pred)*normalized_tensor
        sum = torch.sum(log_lossa, dim=0)
        maskc = torch.ones(y.shape[0],1)
        maskc = maskc.to(device)
        sum = sum * maskc
        log_sum = torch.log(sum).reshape(-1, 1)
        # log_lossa = log_lossa[:, 0]
        # log_lossa = torch.log(log_lossa).reshape(-1, 1)



        log_loss = torch.exp(risk_pred) * mask * normalized_tensor
        log_loss = torch.sum(log_loss, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)

        min_value = 1e-15  # 你可以根据实际情况选择一个合适的小值
        logits_clipped = torch.clamp(log_loss, min=min_value)

        min = (logits_clipped - log_loss)
        nan_mask = torch.isnan(min)






        neg_log_loss = -torch.sum(min * e) / torch.sum(e)
        l2_loss = self.reg(model)

        return neg_log_loss + l2_loss
# class PartialNLL(torch.nn.Module):
#     def __init__(self, config):
#         super(PartialNLL, self).__init__()
#         self.L2_reg = config['l2_reg']
#         self.reg = Regularization(order=2, weight_decay=self.L2_reg)
#
#     # Use matrix to avoid nested loops, for faster calculation
#     def _make_R_rev(self, y):
#         device = y.device
#         R = np.zeros((y.shape[0], y.shape[0]))
#         y = y.cpu().detach().numpy()
#         for i in range(y.shape[0]):
#             idx = np.where(y >= y[i])
#             R[i, idx] = 1
#         return torch.tensor(R, device=device)
#
#
#
#     def forward(self, theta, time, observed, model):
#         R_rev = self._make_R_rev(time).to(theta.device)
#
#         # 首先对所有的 t 找到他的R_rev 矩阵
#
#         # 所有比自己小的都是0， 比自己大的都是1
#
#         # 因此 生存时间比自己少的 就可以忽略不管了
#
#         # 成功实现 ti > tj 的情况
#
#
#         exp_theta = torch.exp(theta)
#         num_observed = torch.sum(observed)
#         loss = -torch.sum((theta.reshape(-1) - torch.log(torch.sum((exp_theta * R_rev.t()), 0))) * observed) / num_observed
#         if np.isnan(loss.data.tolist()):
#             for a, b in zip(theta, exp_theta):
#                 print(a, b)
#         l2_loss = self.reg(model)
#         return loss+l2_loss # 返回的是交叉熵的损失


