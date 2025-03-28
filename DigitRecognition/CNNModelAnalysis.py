# CNN model analysis methods: confusion matrix, layer output, layer weights
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.YlGn):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #print(cm[:, :])
    #plt.figure(figsize = (10,10))
    #plt.imshow(cm)
    
    fig, ax = plt.subplots() #plt.subplots(figsize=(3, 3))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.4)
    ax.figure.colorbar(im, cax=cax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def show_layer_output(nnet, X_sample, layers, conv1d = False):
    for layer in range(len(layers)):
        outputs = []
        reg = nnet.nnet[layer*2].register_forward_hook(
        lambda self, i, o: outputs.append(o))
        nnet.use(X_sample)
        reg.remove()
        output = outputs[0]
        
        n_units = output.shape[1]
        nplots = int(np.sqrt(n_units)) + 1
        fig, ax = plt.subplots(1, n_units, figsize=(20,2))
        axn = ax.flatten()
        fig.suptitle(f'Layer {layer}')
        for axes, unit in zip(axn, range(n_units)):
        #for unit in range(n_units):
            #axes.subplot(nplots, nplots, unit+1)
            if conv1d is True:
                #axes.imshow(output[0, unit, :].reshape(1, -1).detach().cpu().numpy(), cmap='gist_rainbow', aspect = 'auto')
                axes.plot(output[0, unit, :].reshape(-1).detach().cpu().numpy())
            else:
                axes.plot(output[0, unit, :, :].detach().cpu().numpy())
                #axes.imshow(output[0, unit, :, :].detach().cpu().numpy(), cmap='gist_rainbow', aspect = 'auto')
            axes.axis('off')
        #return output

def show_layer_weights(nnet, layers, ch = 0, conv1d = False):
    weights = []
    for layer in range(len(layers)):
        W = nnet.nnet[layer*2].weight.detach()
        weights.append(W)
        #if conv1d is True:
        #    W = nnet.nnet[layer*2].weight.detach()
        #else:
        #    W = nnet.nnet[layer*2].weight.detach()
        n_units = W.shape[0]
        nplots = int(np.sqrt(n_units)) + 1
        fig, ax = plt.subplots(1, n_units, figsize=(20,2))
        axn = ax.flatten()
        fig.suptitle(f'Layer {layer}')
        for axes, unit in zip(axn, range(n_units)):
            #plt.subplot(nplots, nplots, unit + 1)
            if conv1d is True:
                #axes.imshow(W[unit, 0, :].reshape(1, -1).cpu().numpy(), cmap='gist_rainbow', aspect = 'auto')
                axes.plot(W[unit, ch, :].reshape(-1).cpu().numpy())
            else:
                axes.plot(W[unit, ch, :, :].cpu().numpy())
                #axes.imshow(W[unit, 0, :, :].cpu().numpy(), cmap='gist_rainbow', aspect = 'auto')
            axes.axis('off')
    return weights
