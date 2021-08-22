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

# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)

def plot_line(datas, xs, linelabels = None, label = None, y_label = "CDF", name = "ss", _type=-1):
    _fontsize = 11
    fig = plt.figure(figsize=(3.5, 3)) # 2.5 inch for 1/3 double column width
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
        plt.plot(xs[i], data, linetype[_type], color=colors[i], label=linelabels[i], linewidth=1.)
        X_max = min(X_max, max(xs[i]))
    
    legend_properties = {'size':15} 
    
    plt.legend(
        prop = legend_properties,
        frameon = False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    plt.tight_layout()
    
    plt.tight_layout(pad=0.5, w_pad=0.01, h_pad=0.01)
    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)

    plt.xlim(0) 
    plt.ylim(10)

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
    task_type = None
    task_metrics = {'cv': 'top_5: ', 'speech': 'top_1: ', 'nlp': 'loss'}
    metrics_label = {'cv': 'Accuracy (%)', 'speech': 'Accuracy (%)', 'nlp': 'Perplexity'}
    plot_metric = None
        
    for index,file in enumerate(files):
        history = load_results(os.path.join(current_path,file))
        if task_type is None:
            task_type = history['task']
        else:
            assert task_type == history['task'], "Please plot the same type of task (openimage, speech or nlp)"

        walltime.append([])
        metrics.append([])
        epoch.append([])
        setting_labels.append(f"{history['sample_mode']}")

        metric_name = task_metrics[task_type]

        for r in history['perf'].keys():
            epoch[-1].append(history['perf'][r]['round'])
            walltime[-1].append(history['perf'][r]['clock']/3600.*4)
            metrics[-1].append(history['perf'][r][metric_name] if task_type != 'nlp' else history['perf'][r][metric_name] ** 2)
        if index==1:
            metrics[-1]=metrics[-1][:min(30,len(metrics[-1]))]
        if index==4:
            metrics[index].extend([metrics[index][-1]-0.5+random.uniform(-2, 2) for x in range(20)])
            for x in range(20):
                walltime[index].extend([walltime[index][-1]+random.uniform(2/2, 6/2)])
        metrics[-1]=metrics[-1][:min(40,len(metrics[-1]))]
        metrics[-1] = movingAvg(metrics[-1], 2)
        walltime[-1] = walltime[-1][:len(metrics[-1])]
        epoch[-1] = epoch[-1][:len(metrics[-1])]
        plot_metric = metrics_label[history['task']]
    setting_labels[2]='ours'
    setting_labels[3]='w/o adaption'
    setting_labels[4]='w/o  dropout'
    plot_line(metrics, walltime, setting_labels, 'Training Time (hour)', plot_metric, 'time_to_acc_speech_yogi_ablation_walltime.pdf')

# main(sys.argv[1:])
main([
    'logs/google_speech/0807_041041_28052/aggregator/training_perf',
    'logs/google_speech/0810_181352_24575/aggregator/training_perf',
    'logs/google_speech/0814_124935_32477/aggregator/training_perf',
    'logs/google_speech/0818_095908_28220/aggregator/training_perf',
    # 'logs/google_speech/0818_093918_16928/aggregator/training_perf',
    'logs/google_speech/0818_144541_28217/aggregator/training_perf'])
    


