#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 03:54:27 2021.

@author: kpinitas
"""

import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

# color pallette
plt.style.use("seaborn-colorblind")

if not os.path.exists('figures/'):
    os.mkdir('figures/')

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "font.weight": "bold",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 16,
    "font.size": 16,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 2.5,
    "lines.markersize": 20.0,
    "lines.linewidth": 3.5,
    "xtick.major.width": 2.2,
    "ytick.major.width": 2.2,
    "axes.spines.right": False,
    "axes.spines.top": False
}

plt.rcParams.update(tex_fonts)


def set_size(width, fraction=1):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
        Document textwidth or columnwidth in pts
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
        Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def plot_fig(key, k, data, fig_counter):
    """
    Plot figures found in Pinitas et al.

    Parameters
    ----------
    key : str
        DESCRIPTION.
    k : str
        DESCRIPTION.
    data : dict
        DESCRIPTION.
    fig_counter : int
        DESCRIPTION.

    Returns
    -------
    fig_counter : int
        The figure number in an asceding order.
    sv : TYPE
        DESCRIPTION.

    """
    sv = False

    if key == 'classification':
        if k == 'alpha':
            plt.xlabel(r"$a_0$")
            plt.ylabel('Top-1 Accuracy')
            x = data[key][k]['alphas'][0]
            y = data[key][k]['accs'][0]
            plt.plot(x, y)
            fig_counter = fig_counter + 1
            sv = True
        elif k == 'dendrites':
            plt.xlabel('number of units per map')
            plt.ylabel('Top-1 Accuracy')
            x = data[key][k]['dendrites']
            y1 = data[key][k]['dendsom'][0]
            y2 = data[key][k]['som'][0]
            plt.plot(x, y1)
            plt.plot(x, y2)
            fig_counter = fig_counter + 1
            legend = plt.legend(['DendSOM', 'SOM'], frameon=False)
            legend.get_frame().set_facecolor('none')
            sv = True
        elif k == 'rf_size':
            plt.xlabel('receptive field size')
            plt.ylabel('Top-1 Accuracy')
            x = np.array(data[key][k]['ptchs'])[:, 0].tolist()
            y = data[key][k]['accs'][0]
            plt.plot(x, y)
            fig_counter = fig_counter + 1
            sv = True
    elif key == 'continual':
        if k == 'a_crit':
            plt.xlabel(r"$a_{crit}$")
            plt.ylabel('Top-1 Accuracy')
            x = np.log(data[key][k]['ac'][0])
            y1 = data[key][k]['cifar'][0]
            y2 = data[key][k]['mnist'][0]
            plt.plot(x, y1, '.-')
            plt.plot(x, y2, '.-')
            plt.ylim([0.1, 1.0])
            legend = plt.legend(['CIFAR-10', 'MNIST'], frameon=False)
            legend.get_frame().set_facecolor('none')
            sv = True
            fig_counter = fig_counter + 1
        elif k == 'r_exp':
            plt.xlabel(r"$r_{exp}$")
            plt.ylabel('Top-1 Accuracy')
            x = data[key][k]['rx'][0]
            y1 = data[key][k]['cifar'][0]
            y2 = data[key][k]['mnist'][0]
            plt.plot(x, y1, '.-')
            plt.plot(x, y2, '.-')
            plt.ylim([0.1, 1.0])
            legend = plt.legend(['CIFAR-10', 'MNIST'], frameon=False)
            legend.get_frame().set_facecolor('none')
            fig_counter = fig_counter + 1
            sv = True
    elif key == 'general_decay':
        lg = [r"$\lambda=" + str("{:.0e}".format(d)) + r"$" for d in data[key]['lambda']]
        x = data[key]['t']
        if k == 'alpha' or k == 'sigma':
            y = data[key][k]
            if k == 'alpha':
                plt.ylabel(r"$a(t)$")
            else:
                plt.ylabel(r"$\sigma (t)$")
            plt.xlabel('training step')
            for ii in range(len(y)):
                plt.plot(x, y[ii])
            plt.legend(lg, frameon=False)
            fig_counter = fig_counter + 1
            sv = True
    elif key == 'general_nf':
        lg = [r"$t=" + str(d) + r"$" for d in data[key]['t']]
        x = data[key]['d']
        if k == 'h':
            y = data[key]['h']
            plt.ylabel(r"$h(t)$")
            plt.xlabel('distance from BMU')
            plt.ylim([-0.05, 1.22])

            for ii in range(len(y)):
                plt.plot(x, y[ii])
            plt.legend(lg, frameon=False, loc='upper left', ncol=2,
                       mode="expand")
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
PATH = 'figures/'
EXT = 'pdf'

# Load the data
with open(r"fig_data.pickle", "rb") as input_file:
    data = pickle.load(input_file)

fig_counter = 0
for key in data.keys():
    for k in data[key].keys():
        fig = plt.figure(num=fig_counter,
                         figsize=set_size(width=345, fraction=1.0))

        fig_counter, sv = plot_fig(key, k, data, fig_counter)
        if sv:
            sname = f"{PATH}{key}_{k}.{EXT}"
            fig.savefig(sname, dpi=300, format=EXT, bbox_inches='tight')


# Load the data
with open(r"performance_data.pickle", "rb") as handle:
    performance_data = pickle.load(handle)

# fig_counter = 0
for key in performance_data.keys():
    for k in performance_data[key].keys():
        fig = plt.figure(num=fig_counter,
                         figsize=set_size(width=345, fraction=1.0))

        fig_counter, sv = plot_performance_fig(key, k, performance_data,
                                               fig_counter)
        if sv:
            sname = f"{PATH}{key}_{k}.{EXT}"
            fig.savefig(sname, dpi=300)
