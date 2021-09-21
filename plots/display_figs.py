#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 03:54:27 2021

@author: kpinitas
"""
from matplotlib import pyplot as plt
import pickle
from matplotlib import pyplot as mp
import numpy as np 

with open(r"fig_data.pickle", "rb") as input_file:
  data = pickle.load(input_file)
 
    
fig_counter=0

def plot_fig(key,k,data, fig_counter):
    if key=='classification':
        if k=='alpha':
            plt.title('Unsupervised Classification (MNIST)')
            plt.xlabel('a_0')
            plt.ylabel('Top-1 Accuracy')
            x=data[key][k]['alphas'][0]
            y=data[key][k]['accs'][0]
            plt.plot(x,y)
            fig_counter=fig_counter+1
        elif k=='dendrites':
            plt.title('Unsupervised Classification (MNIST)')
            plt.xlabel('number of units per map')
            plt.ylabel('Top-1 Accuracy')
            x=data[key][k]['dendrites']
            y1=data[key][k]['dendsom'][0]
            y2=data[key][k]['som'][0]
            plt.plot(x,y1)
            plt.plot(x,y2)
            fig_counter=fig_counter+1
            plt.legend(['DendSOM','SOM'])
        elif k=='rf_size':
            plt.title('Unsupervised Classification (MNIST)')
            plt.xlabel('receptive field size')
            plt.ylabel('Top-1 Accuracy')
            x=np.array(data[key][k]['ptchs'])[:,0].tolist()
            y=data[key][k]['accs'][0]
            plt.plot(x,y)
            fig_counter=fig_counter+1
    elif key=='continual':
        if k=='a_crit':
            plt.title('Class-IL (Split-Protocol)')
            plt.xlabel('a_crit')
            plt.ylabel('Top-1 Accuracy')
            x=np.log(data[key][k]['ac'][0])
            y1=data[key][k]['cifar'][0]
            y2=data[key][k]['mnist'][0]
            plt.scatter(x,y1)
            plt.scatter(x,y2)
            plt.legend(['CIFAR-10','MNIST'])
            fig_counter=fig_counter+1
        elif k=='r_exp':
            plt.title('Class-IL (Split-Protocol)')
            plt.xlabel('r_exp')
            plt.ylabel('Top-1 Accuracy')
            x=data[key][k]['rx'][0]
            y1=data[key][k]['cifar'][0]
            y2=data[key][k]['mnist'][0]
            plt.scatter(x,y1)
            plt.scatter(x,y2)
            plt.legend(['CIFAR-10','MNIST'])
            fig_counter=fig_counter+1
    elif key=='general_decay':
        x=data[key]['t']
        if k=='alpha'or k=='sigma':
            y =data[key][k]
            if k=='alpha':
                plt.title('Learning rate decay')
                plt.ylabel('learning rate')
            else:
                plt.title('Neighbourhood radius decay')
                plt.ylabel('neighbourhood radius')
            plt.xlabel('training step')
            for ii in range(len(y)):
                plt.plot(x,y[ii])
            plt.legend(data[key]['lambda'])
            fig_counter=fig_counter+1
    elif key=='general_nf':
        x=data[key]['d']
        if k== 'h':
            y =data[key]['h']
            plt.title('Neighbourhood function decay')
            plt.ylabel('h(t)')
            plt.xlabel('distance from BMU')
            for ii in range(len(y)):
                plt.plot(x,y[ii])
            plt.legend(data[key]['t'])
            fig_counter=fig_counter+1
                
        
        
            
        
        
    return fig_counter





for key in data.keys():
    for k in data[key].keys():
        plt.figure(fig_counter)
        fig_counter= plot_fig(key,k,data, fig_counter)
           