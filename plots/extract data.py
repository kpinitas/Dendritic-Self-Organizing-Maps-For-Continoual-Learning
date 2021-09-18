#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:44:13 2021

@author: kpinitas
"""
import pickle
from scipy.io import loadmat, savemat
from matplotlib import pyplot as mp
import numpy as np 
t=np.array(list(range(60000)))
def nf(x, sig):
    return np.exp(-np.power(x -0, 2.) / (2 * np.power(sig, 2.)))
def exp(t,lam):
    return np.exp(-t/lam).tolist()
aa = []
lmbda=[10e1,10e2,10e3]
for l in lmbda:
    aa.append(exp(t,l))

figs={}
sv=loadmat('alphas.mat')

figs['classification']={}
figs['classification']['alpha']={'accs': sv['accs'], 'alphas': sv['alphas']}
sv=loadmat('expand.mat')

figs['continual']={'a_crit': {'ac':sv['ac'],'cifar':sv['cifar'],'mnist':sv['mnist'] }}
sv=loadmat('rx.mat')
figs['continual']['r_exp']= {'rx':sv['rx'],'cifar':sv['cifar'],'mnist':sv['mnist'] }
sv=loadmat('ptchs.mat')
figs['classification']['rf_size']= {'ptchs':sv['ptchs'], 'accs':sv['accs']}
sv=loadmat('sigmas.mat')
figs['general_decay']={'t':t.tolist(),'lambda':lmbda,'alpha':aa,'sigma':aa}
az=aa[1]
ids = [0,9,99,999,2000]
az =list(map(az.__getitem__, ids))
ht=[]

x_values = np.linspace(-3,3, 120)
for sig in az:
    nn=nf(x_values, sig)
    ht.append(nn.tolist())
    mp.plot(x_values,nn)

mp.show()

figs['general_nf']={'t':ids, 'd':x_values.tolist(), 'h':ht}

sv=loadmat('dnds')

figs['classification']['dendrites']={'dendrites':(sv['dnds'][0]**2).tolist(),'som': sv['accs'].tolist()}
sv=loadmat('dnds2')
figs['classification']['dendrites']['dendsom']=sv['accs'].tolist()
with open('fig_data.pickle', 'wb') as handle:
        pickle.dump(figs, handle, protocol=pickle.HIGHEST_PROTOCOL)#

