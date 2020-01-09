# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:44:11 2019

@author: Andreas Nilausen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data load function
def loadevaluationdata(filepath):
    data = pd.read_csv(filepath)
    return data

#Defining data variable
data = loadevaluationdata(r"C:\Users\Andreas Nilausen\Desktop\DTU\Kurser\Introduktion til intelligente systemer\Eksamensprojekt\evaluation\evaluationdata\acc_loss.csv")

#Finding column names
columnNames = list(data.columns.values)
print(columnNames)

#Finding scores (accuracy)
scores = data.val_acc


#Plotting histogram
scores.plot(kind="hist", x = "Accuracy")
plt.xlabel("Accuracy")
plt.title("GAUSSIAN DISTRIBUTION OF VAL. ACC. (..hopefully)")
plt.style.use("grayscale")
plt.show()