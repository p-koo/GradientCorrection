import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 


def plot_improvement(attribution, attribution_corrected, x_min, x_max, y_min, y_max, labelsize=15, fontsize=15, alpha=0.9, s=30):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.add_patch(patches.Rectangle((x_min, 0), 1.3, y_max, facecolor = 'green', fill=True , alpha=0.1))
    ax1.add_patch(patches.Rectangle((x_min, 0), 1.3, y_min, facecolor = 'red', fill=True , alpha=0.1))

    #Cosine
    ax1.scatter(attribution['shallow_relu'], attribution_corrected['shallow_relu']-attribution['shallow_relu'], s=s, edgecolors='g',facecolors='none', marker="o", alpha=alpha) #label='COSINE'
    mean1a=np.average(attribution_corrected['shallow_relu']-attribution['shallow_relu'])

    #Cosine
    ax1.scatter( attribution['shallow_exp'], attribution_corrected['shallow_exp']-attribution['shallow_exp'], s=s, edgecolors='r',facecolors='none', marker="o", alpha=alpha)
    mean1b=np.average(attribution_corrected['shallow_exp']-attribution['shallow_exp'])

    #Cosine
    ax1.scatter( attribution['deep_relu'], attribution_corrected['deep_relu']-attribution['deep_relu'], s=s, edgecolors='b',facecolors='none', marker="o", alpha=alpha)
    mean1c=np.average(np.nan_to_num(attribution_corrected['deep_relu']-attribution['deep_relu']))

    #Cosine
    ax1.scatter( attribution['deep_exp'], attribution_corrected['deep_exp']-attribution['deep_exp'], s=s, edgecolors='black',facecolors='none', marker="o", alpha=alpha)
    mean1d=np.average(np.nan_to_num(attribution_corrected['deep_exp']-attribution['deep_exp']))

    x_ = np.linspace(x_min, 1.0, 100)
    ax1.plot(x_, x_*0, c="black");

    ax1.set_xlim(x_min,1.0)
    ax1.set_ylim(y_min,y_max)
    ax1.tick_params(axis="x", labelsize=labelsize)
    ax1.tick_params(axis="y", labelsize=labelsize)    
    plt.xlabel('Cosine similarity', fontsize=fontsize)
    plt.ylabel('Improvement', fontsize=fontsize)
    fig.tight_layout()  #To prevent filesave cutting of the outside parts of the figure. 

    
def plot_attribution_vs_performance(attribution, performance, labelsize=15, fontsize=15, alpha=0.9, s=30):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    alpha = 0.9
    #---------------------
    #Cosine
    ax1.scatter( attribution['shallow_relu'], performance['shallow_relu'], s=s, edgecolors='g',facecolors='none', marker="o", alpha=alpha) #label='COSINE'

    #---------------------
    #Cosine
    ax1.scatter( attribution['shallow_exp'], performance['shallow_exp'], s=s, edgecolors='r',facecolors='none', marker="o", alpha=alpha)

    #---------------------
    #Cosine
    ax1.scatter( attribution['deep_relu'], performance['deep_relu'], s=s, edgecolors='b',facecolors='none', marker="o", alpha=alpha)

    #---------------------
    #Cosine
    ax1.scatter( attribution['deep_exp'], performance['deep_exp'], s=s, edgecolors='black',facecolors='none', marker="o", alpha=alpha)

    ax1.tick_params(axis="x", labelsize=labelsize)
    ax1.tick_params(axis="y", labelsize=labelsize) 
    plt.xlabel('Cosine similarity', fontsize=fontsize)
    plt.ylabel('Classification AUC', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.tight_layout()  #To prevent filesave cutting of the outside parts of the figure. 


