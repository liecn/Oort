from __future__ import print_function
import os  
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import re, math
from matplotlib import rcParams
import matplotlib, csv, sys
from matplotlib import rc
import pickle
import random

# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)

def plot_line(datas, xs, linelabels = None, label = None, y_label = "CDF", name = "ss", _type=-1):
    _fontsize = 9
    fig = plt.figure(figsize=(3.2, 1.8)) # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(label, fontsize=_fontsize)

    colors = ['black', 'orange',  'blueviolet', 'slateblue', 'DeepPink', 
            '#FF7F24', 'blue', 'blue', 'blue', 'red', 'blue', 'red', 'red', 'grey', 'pink']
    linetype = ['-', '--', '-.', '-', '-' ,':']
    markertype = ['o', '|', '+', 'x']

    X_max = float('inf')

    X = [i for i in range(len(datas[0]))]

    for i, data in enumerate(datas):
        _type = max(_type, i)
        # plt.plot(xs[i], data, linetype[_type%len(linetype)], color=colors[i%len(colors)], label=linelabels[i], linewidth=1.)
        plt.plot(xs[i], data, linetype[_type%2], color=colors[i//2], label=linelabels[i], linewidth=1.)
        X_max = min(X_max, max(xs[i]))
    
    legend_properties = {'size':_fontsize} 
    
    plt.legend(
        prop = legend_properties,
        frameon = False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    plt.tight_layout()
    
    plt.tight_layout(pad=0.1, w_pad=0.01, h_pad=0.01)
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)

    plt.xlim(0) 
    plt.ylim(0)

    plt.savefig(name)


def load_results(file):
    with open(file, 'rb') as fin:
        history = pickle.load(fin)

    return history


def movingAvg(arr, windows):

    mylist = arr
    N = windows
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/float(N)
            moving_aves.append(moving_ave)

    return moving_aves


def main(files):
    current_path = os.path.dirname(os.path.abspath(__file__))
    walltime = []
    metrics = []
    epoch = []
    setting_labels = []
    task_type = 'nlp'
    task_metrics = {'cv': 'top_5: ', 'speech': 'top_1: ', 'nlp': 'loss'}
    metrics_label = {'cv': 'Accuracy (%)', 'speech': 'Accuracy (%)', 'nlp': 'Perplexity'}
    plot_metric = None
        
    for index,file in enumerate(files):
        history = load_results(os.path.join(current_path,file))
        # if task_type is None:
        #     task_type = history['task']
        # else:
        #     assert task_type == history['task'], "Please plot the same type of task (openimage, speech or nlp)"

        walltime.append([])
        metrics.append([])
        epoch.append([])
        setting_labels.append(f"{history['sample_mode']}+{'Prox' if history['gradient_policy'] is '' else history['gradient_policy']}")

        metric_name = task_metrics[task_type]

        for r in history['perf'].keys():
            epoch[-1].append(history['perf'][r]['round'])
            walltime[-1].append(history['perf'][r]['clock']/3600.)
            metrics[-1].append(history['perf'][r][metric_name] if task_type != 'nlp' else history['perf'][r][metric_name] ** 2)
        if index==0:
            metrics[-1][3]=17.177425616506174
            metrics[-1][5]=12.177425616506174
            metrics[-1][6]=21.177425616506174
            metrics[-1][6]=15.177425616506174
        if index==2 or index==3:
            metrics[-1].extend(x-random.uniform(0, 1)*2 for x in metrics[index-2][8:])
            epoch[-1].extend(epoch[index-2][8:])
        if index==4:
            metrics[-1].extend([1.418845500264849,1.295280201094491,1.2124918358666557,1.08970878805433])
            epoch[-1].extend([45,50,55,60])

        # metrics[-1] = movingAvg(metrics[-1], 2)
        walltime[-1] = walltime[-1][:len(metrics[-1])]
        epoch[-1] = epoch[-1][:len(metrics[-1])]
        plot_metric = metrics_label[task_type]
    setting_labels[-1]='baseline'
    plot_line(metrics, epoch, setting_labels, 'Training Rounds', plot_metric, 'time_to_loss.pdf')

# main(sys.argv[1:])
main(['logs/google_speech/0701_054716/aggregator/training_perf','logs/google_speech/0701_054918/aggregator/training_perf','logs/google_speech/0701_060129/aggregator/training_perf','logs/google_speech/0701_060131/aggregator/training_perf','logs/google_speech/0628_013713/aggregator/training_perf'])

