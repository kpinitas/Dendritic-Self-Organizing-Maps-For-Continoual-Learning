#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 02:46:51 2021

@author: kpinitas
"""
import helper
import cupy as cp
from DendSOM import DendSOM
from sklearn.metrics import accuracy_score as accuracy
import pickle
import yaml

def init_model(dname,architecture, metric, features):
    if architecture=='dendsom':
        if dname=='mnist':
            dendsom = DendSOM((10,10),3,features,8,8,metric=metric) 
        elif dataset=='fmnist':
            dendsom = DendSOM((8,8),4,features,10,10,metric=metric)
        elif dname=='cifar10':
            dendsom = DendSOM((4,4),2,features,12,12,metric=metric)
        else:
            dendsom=None
    else:
        if dname=='mnist':
            dendsom = DendSOM(features,1,features,21,21,metric=metric) 
        elif dataset=='fmnist':
            dendsom = DendSOM(features,1,features,18,18,metric=metric)
        elif dname=='cifar10':
            dendsom = DendSOM(features,1,features,29,29,metric=metric)
        else:
            dendsom=None
        
    return dendsom

def train_model(mode, dname, dendsom, x_train,y_train):
    if mode == 'classif':
        dendsom.train_classifier(x_train,y_train,a0=0.95,lmbd=10e2,sigma0=None)
    elif mode == 'incr_task':
        a_crit =5e-3 if dname=='mnist' else 5e-5
        dendsom.train_incr_task(x_train,y_train,a0=0.95,lmbd=10e2,sigma0=None,a_crit=a_crit,r_exp=2)
    elif mode == 'incr_class':
        a_crit =5e-3 if dname=='mnist' else 5e-5
        dendsom.train_incr_class(x_train,y_train,a0=0.95,lmbd=10e2,sigma0=None,a_crit=a_crit,r_exp=2)
    elif mode == 'incr_dom':
        a_crit =5e-3 if dname=='mnist' else 5e-5
        dendsom.train_incr_dom(x_train,y_train,a0=0.95,lmbd=10e2,sigma0=None,a_crit=a_crit,r_exp=2)
    else:
        dendsom=None
    return dendsom


datasets = ['mnist', 'cifar10', 'fmnist']
modes = ['classif','incr_dom','incr_class','incr_task']
metrics=['cos','euc']
architectures=['som','dendsom']
logs={}
n_trials=10

print_only=False

if print_only == False:
    for mode in modes:
        logs[mode]={}
        print(mode)
        datasets_used = datasets[:-1] if mode =='incr_class' else [datasets[0]] if mode=='incr_task'or mode=='incr_dom' else datasets
        for dataset in datasets_used: 
            (X_train,y_train), (X_test,y_test) = helper.import_data(dataset)
            print(dataset)
            logs[mode][dataset]={}
            acc=cp.zeros(shape=(n_trials,))
            for architecture in architectures:
                print(architecture)
                logs[mode][dataset][architecture]={}
#                metrics_used =metrics if mode== 'classif'  else ['cos'] if mode!= 'classif' and architecture=='dendsom' else ['euc']  
                for metric in metrics:
                    print(metric)
                    logs[mode][dataset][architecture][metric]={}
                    for trial in range(n_trials):
                        print('trial: '+str(trial+1))
                        
                        dendsom=init_model(dataset, architecture,metric,X_train[0].shape)
                        shuffle=cp.random.permutation(X_train.shape[0]).tolist()
                        dendsom = train_model(mode,dataset,dendsom,X_train[shuffle],y_train[shuffle])
                        dendsom.calculate_pmi()
                        predictions=[]
                        test_labels=[]
                        for idx in range(X_test.shape[0]):
                            task = None if mode != 'incr_task' else (y_test[idx]-y_test[idx]%2)//2
                            pred = dendsom.classify(X_test[idx],task=task)
                            predictions.append(pred)
                            yt = y_test[idx]%2 if mode=='incr_task' or mode=='incr_dom' else y_test[idx]
                            test_labels.append(yt)
                        
                        acc[trial] = accuracy(predictions, test_labels)
                        print(acc[trial])
                    logs[mode][dataset][architecture][metric]['mean'] = round(cp.mean(acc).tolist(),4)*100
                    logs[mode][dataset][architecture][metric]['std'] = round(cp.std(acc).tolist(),4)*100        
                            
    with open('logs.pickle', 'wb') as handle:
        pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)


                    
with open('logs.pickle', 'rb') as handle:
    logs = pickle.load(handle)


print(yaml.dump(logs, default_flow_style=False))        
        
