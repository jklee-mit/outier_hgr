# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:22:43 2020

@author: jklee
"""

import torch
import torch.nn as nn
import scipy.linalg as la
import numpy as np

def max_corr_obj_cont_disc(f,g,data_x,data_y):
    """Computes soft-HGR objective for continuous x and discrete y"""
    num_samps = data_x.shape[0]
    
    outputs_f = f(data_x)
    outputs_f -= outputs_f.mean(dim=0)
    cov_f = torch.mm(torch.t(outputs_f),outputs_f)/(num_samps-1)
    
    outputs_g = g[data_y.long(),:]
    outputs_g -= outputs_g.mean(dim=0)
    cov_g = torch.mm(torch.t(outputs_g),outputs_g)/(num_samps-1)
    loss = torch.trace(torch.mm(torch.t(outputs_f),outputs_g.float())/(num_samps-1)) - 0.5*torch.trace(torch.mm(cov_f,cov_g.float()))
    
    return loss

def get_std_devs(net,inputs):
    outputs = net(inputs)
    outputs -= outputs.mean(dim=0)
    stds = torch.sqrt(torch.diag(torch.mm(outputs.permute(1,0),outputs)))
    stds[stds==0] = 1
    return stds

def compute_max_corr_func(net,data,labels,num_classes):
    """Computes maximal correlation and also returns the associated g(y)"""
    outputs = net(data)
    outputs -= outputs.mean(dim=0)
    outputs /= get_std_devs(net,data)
    g_y = torch.zeros((num_classes,outputs.shape[1]))
    g_y = g_y.cuda()
    for idx,row in enumerate(outputs.split(1)):
        g_y[labels[idx]] += row.detach().reshape(-1)
    g_y/=labels.shape[0]
    
    p_y = (1/num_classes)*np.ones((num_classes,1))
    g_y = g_y.detach().cpu().numpy()
    q,r = la.qr(g_y * np.sqrt(p_y),mode='economic')
    # r = la.cholesky(norm_mat(g_y, y, N))
    g_y =  np.dot(g_y,la.inv(r))
    g_y = torch.tensor(g_y).cuda()
    
    sigma = torch.zeros(outputs.shape[1])
    sigma = sigma.cuda()
    for idx,row in enumerate(outputs.split(1)):
        sigma += row.detach().reshape(-1)*g_y[labels[idx],:]
    sigma/=labels.shape[0]
    #make sure signs are positive
    g_y *= sigma.sign()
    sigma *= sigma.sign()
    return sigma, g_y

class max_corr_net(nn.Module):
    """
    class for f-netowrks for hgr
    """
    def __init__(self,num_features_1,num_features_2, num_classes):
        super(max_corr_net, self).__init__()
        self.fc1 = nn.Linear(num_features_1, num_features_2)
        self.fc2 = nn.Linear(num_features_2, num_classes)
    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        return output

def weighted_network_output(net,sigma,g,inputs):
    """Output weighted sum(sigma*f*g) for both values of g"""
    outputs = net(inputs)
    outputs -= outputs.mean(dim=0)
    outputs *= sigma.reshape(1,-1)
    outputs= torch.mm(outputs, g.permute(1,0))
    return outputs