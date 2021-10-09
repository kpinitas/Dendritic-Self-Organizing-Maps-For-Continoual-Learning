import helper
import cupy as cp
from DendSOM import DendSOM
from sklearn.metrics import accuracy_score as accuracy
import pickle
import yaml


def train_model(mode, dname, dendsom, x_train,y_train,stop_on_task):
    if mode == 'classif':
        dendsom.train_classifier(x_train,y_train,a0=0.95,lmbd=10e2,sigma0=None)
    elif mode == 'incr_task':
        a_crit =5e-3 if dname=='mnist' else 5e-5
        dendsom.train_incr_task(x_train,y_train,a0=0.95,lmbd=10e2,sigma0=None,a_crit=a_crit,r_exp=2,stop_on_task=stop_on_task)
    elif mode == 'incr_class':
        a_crit =5e-3 if dname=='mnist' else 5e-5
        dendsom.train_incr_class(x_train,y_train,a0=0.95,lmbd=10e2,sigma0=None,a_crit=a_crit,r_exp=2,stop_on_task=stop_on_task)
    elif mode == 'incr_dom':
        a_crit =5e-3 if dname=='mnist' else 5e-5
        dendsom.train_incr_dom(x_train,y_train,a0=0.95,lmbd=10e2,sigma0=None,a_crit=a_crit,r_exp=2,stop_on_task=stop_on_task)
    else:
        dendsom=None
    return dendsom


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

datasets = ['cifar10','mnist']
modes = ['incr_dom','incr_class','incr_task']
metrics=['cos','euc']
architectures=['som','dendsom']
stop_on_task=[1,2,3,4,5]
logs={}
n_trials=3


for mode in modes:
    logs[mode]={}
    datasets_used = ['mnist'] if mode!='incr_class' else datasets
    for dataset in datasets_used:
        (X_train,y_train), (X_test,y_test) = helper.import_data(dataset)
        logs[mode][dataset]={}
        for arch in architectures:
            logs[mode][dataset][arch]={}
            metrics_used = ['cos'] if arch=='dendsom' else metrics
            for metric in metrics_used:
                logs[mode][dataset][arch][metric]={}
                logs[mode][dataset][arch][metric]['mean']=[]
                logs[mode][dataset][arch][metric]['std']=[]
                
                for stop in stop_on_task:
                    acc=cp.zeros((n_trials,))
                    for trial in range(n_trials):
                        print('trial: '+str(trial+1))
                        
                        dendsom=init_model(dataset, arch,metric,X_train[0].shape)
                        shuffle=cp.random.permutation(X_train.shape[0]).tolist()
                        dendsom = train_model(mode,dataset,dendsom,X_train[shuffle],y_train[shuffle],stop_on_task=stop)
                        dendsom.calculate_pmi()
                        predictions=[]
                        test_labels=[]
                        if stop!=None:
                            test_x = X_test[y_test<=2*stop-1]
                            test_y =y_test[y_test<=2*stop-1]
                        else:
                            test_x = X_test
                            test_y =y_test
                        for idx in range(test_y.shape[0]):
                            task = None if mode != 'incr_task' else (test_y[idx]-test_y[idx]%2)//2
                            pred = dendsom.classify(test_x[idx],task=task)
                            predictions.append(pred)
                            yt = test_y[idx]%2 if mode=='incr_task' or mode=='incr_dom' else test_y[idx]
                            test_labels.append(yt)
                        
                        acc[trial] = accuracy(predictions, test_labels)
                        print(acc[trial])
                    logs[mode][dataset][arch][metric]['mean'].append(round(cp.mean(acc).tolist(),4)*100)
                    logs[mode][dataset][arch][metric]['std'].append(round(cp.std(acc).tolist(),4)*100)
                    
with open('performance.pickle', 'rb') as handle:
    logs = pickle.load(handle)
