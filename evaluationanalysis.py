#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import seaborn as sns

'''
This function loads the data into a dataframe, extracts the validation
accuracies and converts these into a numpy array
'''
def val_acc(path):
    #Loading data
    df = pd.read_csv(path)
    #Extracting validation accuracies from the dataframe
    df_valacc = df["val_acc"]
    #Converting to numpy array
    valacc = df_valacc.values
    
    return valacc

'''
This function calculates the confidence interval for the different
models (for validation accuracies) and collects them in an array
'''
def confidence(valacc):
    #Sample size is iterrations (1000)
    n = 1000
    
    #Alpha is set to 0.05 (confidence level is chosen at 0.95)
    alpha = 0.05 / 2
    
    #Critical value is calculated (https://machinelearningmastery.com/critical-values-for-statistical-hypothesis-testing/)
    p = 1 - alpha
    critical_value = norm.ppf(p)
    
    #Sample proportion (p-hat) is the mean of all accuracies
    p_hat = np.mean(valacc)
    
    # Calculating confidence interval
    conf_lower = p_hat - critical_value * math.sqrt((p_hat * (1 - p_hat)) / n)
    conf_upper = p_hat + critical_value * math.sqrt((p_hat * (1 - p_hat)) / n)
    
    #Arranging in array
    conf_intvs = np.array([conf_lower, conf_upper])
    
    return conf_intvs

'''
This function loads data into dataframe and extracts train accuracies and validation
accuracies of A_to_E model and saves these.
'''
def accuracyplot(path):
    #Loading data to csv file
    df = pd.read_csv(path)

    #Extracting validation accuracies from the dataframe
    df_acc = df["acc"]
    df_valacc = df["val_acc"]

    #Converting to numpy array
    acc = df_acc.values
    valacc = df_valacc.values

    #Setting font size
    ax = plt.subplot(111, xlabel='Epochs', ylabel='Accuracy')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

    #Plotting both graphs
    plt.plot(acc)
    plt.plot(valacc)

    #Plotting both end points
    plt.plot(80, 0.981, 'bo')
    plt.plot(80, 0.709677, 'ro')

    #Plot settings
    plt.style.use('seaborn-darkgrid')
    plt.legend(["Train accuracy", "Test accuracy", "0.98", "0.71"])
    plt.savefig("ae2_accuracy")
    plt.show()
    
    return

'''
This function creates a bar chart a specific array of validation accuracies
and saves these as a .png file. 
'''
def histogram(valacc):
    #Setting font size
    ax = plt.subplot(111, xlabel='x', ylabel='y')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    
    #Plot settings
    plt.hist(valacc, bins=30, range=(0, 1))
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.savefig("besthistogram.png")
    plt.style.use("seaborn-darkgrid")
    
    #Showing plot
    plt.show()
    
    return

'''
This function creates a density plot with six different datasets consisting
of validation accuracies and saves these as a .png file.
'''
def densityplot(valacc1, valacc2, valacc3, valacc4, valacc5, valacc6):
    
    ax = plt.subplot(111, xlabel='Accuracy', ylabel='Density')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

    sns.kdeplot(valacc1, label="base", clip=(0.0, 1.0))
    sns.kdeplot(valacc2, label="noAug", clip=(0.0, 1.0))
    sns.kdeplot(valacc1, label="avgPool", clip=(0.0, 1.0))
    sns.kdeplot(valacc1, label="adam", clip=(0.0, 1.0))
    sns.kdeplot(valacc1, label="adadel", clip=(0.0, 1.0))
    sns.kdeplot(valacc1, label="combi", clip=(0.0, 1.0))
    plt.savefig("densityplot")
    
    plt.legend()
    return