"""This program is to predict the class for the input time step given.
    10 time step has to be given as the input to the model loaded"""

# Importing the required packages
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input

# Function to load the current working directory


def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
        print(current_path)
    return current_path


# Loding the path of the different files required to the variables
Folder = os.path.join(get_parent_dir(1), "Process_Identification")
Data_Folder = os.path.join(Folder, "Dataset")
data_dir = os.path.join(Data_Folder, "output_random_tasks.csv")
Model_Data_Folder = os.path.join(Folder, "Model_Data")
saved_model = Model_Data_Folder

# Reading the dataframe from the test file
df = pd.read_csv(data_dir)

# Loading the model
model = load_model(os.path.join(saved_model, "trained_21_02_2021.h5"))

#List for final output
output = []

# Function for generating time-series of 10 steps
def rows_generator(df):
    i = 0
    while (i+10) <= df.shape[0]:
        yield df.iloc[i:(i+10):1, :]
        i += 10

i = 1
# Iteration over each time-Series
for df in rows_generator(df): 
    print(f'Time-series #{i}')
    print(df)
    
    # Converting the input file to the array format
    data = np.array(df, dtype=np.float32)
    
    # Formating the shape of the input to the shape required by trained model
    data = np.reshape(data, (-1, 10, 13))
    
    # Giving input for prediction
    y = model.predict(data)
    
    # converting tensor output array to numpy array format
    y = np.array(y)
    y = y[:, :]
    print(y)
    
    # Printing the predicting the class based on the range specified.
    if (y > 0) & (y < 1.5):
        print(y, "The process is: Machining")
        output.extend((y,"The process is: Machining"))
    elif (y > 1) & (y < 2.5):
        print(y, "The process is:Drawer Open")
        output.extend((y,"The process is:Drawer Open"))
    elif (y > 2.5) & (y < 3.5):
        print(y, "The process is: Drawe Close")
        output.extend((y, "The process is: Drawe Close"))
    else:
        print('none')
        output.append('none')
        
    i += 1
    
# Generating process identification result file
output_df = pd.DataFrame(output)
file_name = os.path.basename(data_dir)
file_name =(os.path.splitext(file_name)[0])
output_df.to_csv("/content/drive/MyDrive/Object_detection_and_Process_identification/Process_Identification/{}_{}.csv".format("final_output",file_name))