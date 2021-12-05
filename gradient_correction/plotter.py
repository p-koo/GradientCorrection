import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 


def plot_improvement(attribution, attribution_corrected, x_min, x_max, y_min, y_max, x_label, labelsize=15, fontsize=15, alpha=0.9, s=30):

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
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel('Improvement', fontsize=fontsize)
    fig.tight_layout()  #To prevent filesave cutting of the outside parts of the figure. 

    
def plot_attribution_vs_performance(attribution, performance, x_label, labelsize=15, fontsize=15, alpha=0.9, s=30):

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
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel('Classification AUC', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.tight_layout()  #To prevent filesave cutting of the outside parts of the figure. 


def plot_regularization(experiment_name,log_reg, Performance, Cosine, Performance_std, Cosine_std, y1_min, y1_max, y2_min, y2_max, labelsize=20, fontsize=25):
    # Plot Line1 (Left Y Axis)
    fig, ax1 = plt.subplots(1,1,figsize=(12,7), dpi= 80)
    ax1.plot(log_reg, Performance, color='tab:red', linewidth=5.0)
    ax1.scatter(log_reg, Performance, color='tab:red', s=100)
    ax1.fill_between(log_reg, np.array(Performance)-np.array(Performance_std), np.array(Performance)+np.array(Performance_std), color="r", alpha=0.2)
    ax1.set_ylim(y1_min,y1_max)

    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(log_reg, Cosine, color='tab:blue', linewidth=5.0)
    ax2.scatter(log_reg, Cosine, color='tab:blue', s=100)
    ax2.fill_between(log_reg, np.array(Cosine)-np.array(Cosine_std), np.array(Cosine)+np.array(Cosine_std), color="b", alpha=0.2)
    ax2.set_ylim(y2_min,y2_max)

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel('Angle regularization strength (log)', fontsize=fontsize)
    ax1.tick_params(axis='x', rotation=0, labelsize=labelsize)
    ax1.set_ylabel('Classification AUC', color='tab:red', fontsize=fontsize)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' , labelsize=labelsize)
    ax1.grid(alpha=.4)

    # ax2 (right Y axis)
    ax2.set_ylabel("Cosine similarity", color='tab:blue', fontsize=fontsize)
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=labelsize)
    ax2.set_title(experiment_name, fontsize=fontsize)
    fig.tight_layout()


    plt.show()
    
def plot_regularization_angles(log_reg, Angles_std, y_min, y_max, labelsize=20, fontsize=25):
    # Plot Line1 (left Y Axis)
    fig, ax1 = plt.subplots(1,1,figsize=(13.5,3), dpi= 80)
    ax1.plot(log_reg, Angles_std, color='tab:green', linewidth=7.0)
    ax1.set_ylim(y_min,y_max)

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel('Angle regularization strength (log)', fontsize=fontsize)
    ax1.tick_params(axis='x', rotation=0, labelsize=labelsize)
    ax1.set_ylabel('Angle std', color='tab:green', fontsize=fontsize)
    ax1.tick_params(axis='y', rotation=0, labelsize=labelsize)
    ax1.grid(alpha=.4)

    plt.show()
