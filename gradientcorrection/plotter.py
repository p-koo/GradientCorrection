import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 

def plot_improvement(attribution, attribution_corrected, x_min, x_max, y_min, y_max ):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.add_patch( patches.Rectangle( (x_min, 0), 1.3, y_max, facecolor = 'green', fill=True , alpha=0.1 ) )
    ax1.add_patch( patches.Rectangle( (x_min, 0), 1.3, y_min, facecolor = 'red', fill=True , alpha=0.1 ) )

    alpha=0.9
    #---------------------
    #Cosine
    ax1.scatter( attribution['cnn-local_relu'], attribution_corrected['cnn-local_relu']-attribution['cnn-local_relu'], s=30, edgecolors='g',facecolors='none', marker="o", alpha=alpha) #label='COSINE'
    mean1a=np.average(attribution_corrected['cnn-local_relu']-attribution['cnn-local_relu'])

    #---------------------
    #Cosine
    ax1.scatter( attribution['cnn-local_exponential'], attribution_corrected['cnn-local_exponential']-attribution['cnn-local_exponential'], s=30, edgecolors='r',facecolors='none', marker="o", alpha=alpha)
    mean1b=np.average(attribution_corrected['cnn-local_exponential']-attribution['cnn-local_exponential'])

    #---------------------
    #Cosine
    ax1.scatter( attribution['cnn-dist_relu'], attribution_corrected['cnn-dist_relu']-attribution['cnn-dist_relu'], s=30, edgecolors='b',facecolors='none', marker="o", alpha=alpha)
    mean1c=np.average(np.nan_to_num(attribution_corrected['cnn-dist_relu']-attribution['cnn-dist_relu']))

    #---------------------
    #Cosine
    ax1.scatter( attribution['cnn-dist_exponential'], attribution_corrected['cnn-dist_exponential']-attribution['cnn-dist_exponential'], s=30, edgecolors='black',facecolors='none', marker="o", alpha=alpha)
    mean1d=np.average(np.nan_to_num(attribution_corrected['cnn-dist_exponential']-attribution['cnn-dist_exponential']))

    x__ = np.linspace(x_min, 1.0, 100)
    ax1.plot(x__, x__*0, c="black");

    ax1.set_xlim(x_min,1.0)
    ax1.set_ylim(y_min,y_max)
    ax1.tick_params(axis="x", labelsize=15)
    ax1.tick_params(axis="y", labelsize=15)    
    fig.tight_layout()  #To prevent filesave cutting of the outside parts of the figure. 

    plt.xlabel('Cosine similarity', fontsize=15)
    plt.ylabel('Improvement', fontsize=15)
    plt.savefig('drive/My Drive/results/Cosine_Saliency.pdf', bbox_inches='tight')  
    plt.show()  
