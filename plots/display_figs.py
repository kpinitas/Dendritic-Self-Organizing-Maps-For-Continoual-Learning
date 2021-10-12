#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 03:54:27 2021.

@author: kpinitas
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams["figure.autolayout"] = True
mpl.rcParams['axes.linewidth'] = 2.5
mpl.rcParams['lines.linewidth'] = 3.5
mpl.rcParams['xtick.major.width'] = 2.2
mpl.rcParams['ytick.major.width'] = 2.2
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['text.usetex'] = True

plt.style.use("seaborn-colorblind")


def plot_fig(key, k, data, fig_counter):
    """
    Plot figures found in Pinitas et al.

    Parameters
    ----------
    key : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    fig_counter : TYPE
        DESCRIPTION.

    Returns
    -------
    fig_counter : TYPE
        DESCRIPTION.
    sv : TYPE
        DESCRIPTION.

    """
    sv = False

    if key == 'classification':
        if k == 'alpha':
            # plt.title('Unsupervised Classification (MNIST)', fontsize=20)
            plt.xlabel('$a_0$')
            plt.ylabel('Top-1 Accuracy')
            x = data[key][k]['alphas'][0]
            y = data[key][k]['accs'][0]
            plt.plot(x, y)
            fig_counter = fig_counter + 1
            sv = True
        elif k == 'dendrites':
            # plt.title('Unsupervised Classification (MNIST)', fontsize=20)
            plt.xlabel('number of units per map')
            plt.ylabel('Top-1 Accuracy')
            x = data[key][k]['dendrites']
            y1 = data[key][k]['dendsom'][0]
            y2 = data[key][k]['som'][0]
            plt.plot(x, y1)
            plt.plot(x, y2)
            fig_counter = fig_counter + 1
            plt.legend(['DendSOM', 'SOM'], frameon='False')
            sv = True
        elif k == 'rf_size':
            # plt.title('Unsupervised Classification (MNIST)', fontsize=20)
            plt.xlabel('receptive field size')
            plt.ylabel('Top-1 Accuracy')
            x = np.array(data[key][k]['ptchs'])[:, 0].tolist()
            y = data[key][k]['accs'][0]
            plt.plot(x, y)
            fig_counter = fig_counter + 1
            sv = True
    elif key == 'continual':
        if k == 'a_crit':
            # plt.title('Class-IL (Split-Protocol)', fontsize=20)
            plt.xlabel('$a_{crit}$')
            plt.ylabel('Top-1 Accuracy')
            x = np.log(data[key][k]['ac'][0])
            y1 = data[key][k]['cifar'][0]
            y2 = data[key][k]['mnist'][0]
            plt.scatter(x, y1)
            plt.scatter(x, y2)
            plt.legend(['CIFAR-10', 'MNIST'], frameon='False')
            sv = True
            fig_counter = fig_counter + 1
        elif k == 'r_exp':
            # plt.title('Class-IL (Split-Protocol)', fontsize=20)
            plt.xlabel('$r_{exp}$')
            plt.ylabel('Top-1 Accuracy')
            x = data[key][k]['rx'][0]
            y1 = data[key][k]['cifar'][0]
            y2 = data[key][k]['mnist'][0]
            plt.scatter(x, y1)
            plt.scatter(x, y2)
            plt.legend(['CIFAR-10', 'MNIST'], frameon='False')
            fig_counter = fig_counter + 1
            sv = True
    elif key == 'general_decay':
        lg = ['$\lambda = ' + str(int(d)) + '$' for d in data[key]['lambda']]
        x = data[key]['t']
        if k == 'alpha' or k == 'sigma':
            y = data[key][k]
            if k == 'alpha':
                # plt.title('Learning rate decay', fontsize=20)
                plt.ylabel('$a(t)$')
            else:
                # plt.title('Neighbourhood radius decay', fontsize=20)
                plt.ylabel('$\sigma (t)$')
            plt.xlabel('training step')
            for ii in range(len(y)):
                plt.plot(x, y[ii])
            plt.legend(lg, frameon='False')
            fig_counter = fig_counter + 1
            sv = True
    elif key == 'general_nf':
        lg = ['$t=' + str(d) + '$' for d in data[key]['t']]
        x = data[key]['d']
        if k == 'h':
            y = data[key]['h']
            # plt.title('Neighbourhood function decay', fontsize=20)
            plt.ylabel('$h(t)$')
            plt.xlabel('distance from BMU')
            for ii in range(len(y)):
                plt.plot(x, y[ii])
            plt.legend(lg, frameon='False')
            fig_counter = fig_counter + 1
            sv = True

    return fig_counter, sv


def plot_performance_fig(key, k, data, fig_counter):
    """
    Plot CL performance-related figures found in Pinitas et al.

    Parameters
    ----------
    key : string.
    k : string
    data : dict
    fig_counter : int
    Returns
    -------
    fig_counter : int
    sv : boolean
    """
    sv = False
    x = list(range(5))
    if key == 'incr_class':
        if k == 'mnist':
            plt.title('Class-IL (Split-MNIST)', fontsize=20)
        else:
            plt.title('Class-IL (Split-CIFAR-10)', fontsize=20)
    elif key == 'incr_dom':
        plt.title('Domain-IL (Split-MNIST)', fontsize=20)
    else:
        plt.title('Task-IL (Split-MNIST)', fontsize=20)

    plt.xlabel('number of tasks')
    plt.ylabel('Top-1 Accuracy')

    dendsom = data[key][k]['dendsom']['cos']['mean']
    csom = data[key][k]['som']['cos']['mean']
    esom = data[key][k]['som']['euc']['mean']

    plt.plot(x, dendsom)
    plt.plot(x, csom)
    plt.plot(x, esom)
    plt.legend(['$DendSOM$', '$SOM_{cos}$', '$SOM_{euc}$'], frameon=False)
    plt.show()
    fig_counter = fig_counter + 1
    sv = True
    return fig_counter, sv


# Define key parameters
PATH = './figures/'
EXT = '.pdf'

# Load the data
with open(r"fig_data.pickle", "rb") as input_file:
    data = pickle.load(input_file)

fig_counter = 0
for key in data.keys():
    for k in data[key].keys():
        fig = plt.figure(fig_counter)

        fig_counter, sv = plot_fig(key, k, data, fig_counter)
        if sv:
            sname = PATH + key + '_' + k + EXT
            fig.savefig(sname, dpi=300)


# Load the data
with open(r"performance_data.pickle", "rb") as handle:
    performance_data = pickle.load(handle)

fig_counter = 0
for key in performance_data.keys():
    for k in performance_data[key].keys():
        fig = plt.figure(fig_counter)

        fig_counter, sv = plot_performance_fig(key, k, performance_data,
                                               fig_counter)
        if sv:
            sname = PATH + key + '_' + k + EXT
            fig.savefig(sname, dpi=300)
