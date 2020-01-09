import numpy as np
import pandas as pd
import os

#Data load function
def loadevaluationdata(filepath):
    data = pd.read_csv(filepath)
    return data

#Defining data variable
data = loadevaluationdata(r"C:\Users\Andreas Nilausen\Desktop\DTU\Kurser\Introduktion til intelligente systemer\Eksamensprojekt\evaluation\evaluationdata\acc_loss.csv")

#Finding column names
columnNames = list(data.columns.values)
print(columnNames)

#Plotting histogram
print(data.plot(kind="hist"))


