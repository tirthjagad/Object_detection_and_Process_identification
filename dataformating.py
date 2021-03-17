import pandas as pd
import numpy as np
import os
from os import path, makedirs
import sys

def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
        print(current_path)
    return current_path

Folder = os.path.join(get_parent_dir(1), "Process_Identification")
Data_Folder = os.path.join(Folder, "data_from_yolo")
Model_Data_Folder = os.path.join(Folder, "Model_Data")
Model_Data = Model_Data_Folder

data_dir = os.path.join(
    Data_Folder,
    "output_random_tasks.csv")
    
df = pd.read_csv(data_dir, header=0)
df.drop(df.columns[[0]], axis=1, inplace=True)

# Converting yolo label to training file format
# yolo label= not_empty:0 empty:1 hand:2 tool:3
# process_ identification label= hand:1 tool:2 empty:3 not_empty:4
df["label"].replace({0:4, 1:3, 2:1, 3:2}, inplace=True)

#print(df)
#df.to_csv("/content/drive/MyDrive/Object_detection_and_Process_identification/Process_Identification/Dataset/trial.csv", index=False)
# column 'match' based on True(multi detection) and False(single detection)

# identifying the multiple detection in one time-step
df["match"] = df.image.eq(df.image.shift())
df.drop('confidence', axis=1, inplace=True)

df[['hand_id',
    'x_hand_id',
    'y_hand_id',
    'tool_id',
    'x_tool_id',
    'y_tool_id',
    'empty_id',
    'x_empty_id',
    'y_empty_id',
    'not_empty_id',
    'x_not_empty_id',
    'y_not_empty_id',
    'C',
    'C10',
    'C20',
    'pred_id']] = pd.DataFrame([["",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 "",
                                 ""]],
                               index=df.index)

# For rows 1 to 4 next rows indexes are considered to arrange the same class_id rows


def rows1to4arrange(i):
    if ((df.at[i,'image']) == (df.at[i + 3,'image']) and (df.at[i,'image']) == (df.at[i + 2,'image']) and (df.at[i,'image']) == (df.at[i + 1,'image'])):
        if ((df.at[i, 'label']) == (df.at[i + 3, 'label'])):
            df.at[i + 3, 'match'] = False
        elif ((df.at[i, 'label']) == (df.at[i + 2, 'label'])):
            df.at[i + 2, 'match'] = False
        elif ((df.at[i, 'label']) == (df.at[i + 1, 'label'])):
            df.at[i + 1, 'match'] = False

    elif ((df.at[i, 'image']) == (df.at[i + 2, 'image']) and (df.at[i, 'image']) == (df.at[i + 1, 'image'])):
        if ((df.at[i, 'label']) == (df.at[i + 2, 'label'])):
            df.at[i + 2, 'match'] = False
        elif ((df.at[i, 'label']) == (df.at[i + 1, 'label'])):
            df.at[i + 1, 'match'] = False

    elif ((df.at[i, 'image']) == (df.at[i + 1, 'image'])):
        if ((df.at[i, 'label']) == (df.at[i + 1, 'label'])):
            df.at[i + 1, 'match'] = False

# For class 'hand_id' data to transform into require input format


def label1(valueindex, value):
    df.at[valueindex - value, 'C'] = row['image']
    df.at[valueindex - value, 'x_hand_id'] = row['xmin']
    df.at[valueindex - value, 'y_hand_id'] = row['ymin']
    df.at[valueindex - value, 'hand_id'] = row['label']
    df.at[valueindex - value, 'pred_id'] = 0

# For class 'tool_id' data to transform into require input format


def label2(valueindex, value):
    df.at[valueindex - value, 'C10'] = row['image']
    df.at[valueindex - value, 'x_tool_id'] = row['xmin']
    df.at[valueindex - value, 'y_tool_id'] = row['ymin']
    df.at[valueindex - value, 'tool_id'] = row['label']
    df.at[valueindex - value, 'pred_id'] = 0

# For class 'empty_id' data to transform into require input format


def label3(valueindex, value):
    df.at[valueindex - value, 'image'] = row['image']
    df.at[valueindex - value, 'x_empty_id'] = row['xmin']
    df.at[valueindex - value, 'y_empty_id'] = row['ymin']
    df.at[valueindex - value, 'empty_id'] = row['label']
    df.at[valueindex - value, 'pred_id'] = 0

# For class 'not_empty_id' data to transform into require input format


def label4(valueindex, value):
    df.at[valueindex - value, 'C20'] = row['image']
    df.at[valueindex - value, 'x_not_empty_id'] = row['xmin']
    df.at[valueindex - value, 'y_not_empty_id'] = row['ymin']
    df.at[valueindex - value, 'not_empty_id'] = row['label']
    df.at[valueindex - value, 'pred_id'] = 0

# For only single detection in the image and to add zeros to all other columns


def nodiffernce(valueindex):
    df.at[valueindex, 'image'] = ""
    df.at[valueindex, 'xmin'] = 0
    df.at[valueindex, 'ymin'] = 0
    df.at[valueindex, 'label'] = 0
    df.at[valueindex, 'pred_id'] = 0


# after 4th rows consideration is carried out by comparing current row with previous rows to arrange the same class_id rows
# Maximum 4 detection of object in the one time-step is considered
# If multiple object of same class detected then it moved to next rows
for i, r in df.iterrows():
    if i == 0:
        rows1to4arrange(i)
    elif i == 1:
        rows1to4arrange(i)

    elif i == 2:
        rows1to4arrange(i)

    elif i == 3:
        rows1to4arrange(i)

    elif ((df.at[i, 'image']) == (df.at[i - 3, 'image']) and (df.at[i, 'image']) == (df.at[i - 2, 'image']) and (df.at[i, 'image']) == (df.at[i - 1, 'image'])):
        if ((df.at[i, 'label']) == (df.at[i - 3, 'label'])):
            df.at[i, 'match'] = False
        elif ((df.at[i, 'label']) == (df.at[i - 2, 'label'])):
            df.at[i, 'match'] = False
        elif ((df.at[i, 'label']) == (df.at[i - 1, 'label'])):
            df.at[i, 'match'] = False

    elif ((df.at[i, 'image']) == (df.at[i - 2, 'image']) and (df.at[i, 'image']) == (df.at[i - 1, 'image'])):
        if ((df.at[i, 'label']) == (df.at[i - 2, 'label'])):
            df.at[i, 'match'] = False
        elif ((df.at[i, 'label']) == (df.at[i - 1, 'label'])):
            df.at[i, 'match'] = False

    elif ((df.at[i, 'image']) == (df.at[i - 1, 'image'])):
        if ((df.at[i, 'label']) == (df.at[i - 1, 'label'])):
            df.at[i, 'match'] = False



for index, row in df.iterrows():
    if (row['match']):  # for multiple detection

        if ((df.at[index - 2, 'match']) & (df.at[index - 1, 'match'])):
            if (row['label'] == 1):
                label1(index, 3)

            elif (row['label'] == 2):
                label2(index, 3)

            elif (row['label'] == 4):
                label4(index, 3)

            elif (row['label'] == 3):
                label3(index, 3)

        elif ((df.at[index - 1, 'match'])):  # in previous 3rd row if the same time-step multiple detection is there then it will shift the current row to 1st detection rows with respect to its class id
            if (row['label'] == 1):
                label1(index, 2)

            elif (row['label'] == 2):
                label2(index, 2)

            elif (row['label'] == 4):
                label4(index, 2)

            elif (row['label'] == 3):
                label3(index, 2)
        else:  # in previous 2nd row if the same time-step multiple detection is there then it will shift the current row to 1st detection rows with respect to its class id

            if (row['label'] == 1):
                label1(index, 1)

            elif (row['label'] == 2):
                label2(index, 1)

            elif (row['label'] == 4):
                label4(index, 1)

            elif (row['label'] == 3):
                label3(index, 1)

    else:  # For single detection in the time-step
        if (row['label'] == 1):
            nodiffernce(index)
            label1(index, 0)

        elif (row['label'] == 2):
            nodiffernce(index)
            label2(index, 0)

        elif (row['label'] == 4):
            nodiffernce(index)
            label4(index, 0)

        elif (row['label'] == 3):
            label3(index, 0)


df = df[(df['match'] == False)]  # Consideration of arranged rows only
df = df.apply(
    lambda x: x.str.strip() if isinstance(x, str) else x).replace('', 0)  # to convert values Nan to 0
df.drop(df.columns[[0]], axis=1, inplace=True)
df.drop('match', axis=1, inplace=True)
df.drop('C', axis=1, inplace=True)
df.drop('C10', axis=1, inplace=True)
df.drop('C20', axis=1, inplace=True)
df.drop('xmax', axis=1, inplace=True)
df.drop('ymax', axis=1, inplace=True)
df.drop('xmin', axis=1, inplace=True)
df.drop('ymin', axis=1, inplace=True)
df.drop('label', axis=1, inplace=True)

file_name = os.path.basename(data_dir)
file_name =(os.path.splitext(file_name)[0])
df.to_csv("/content/drive/MyDrive/Object_detection_and_Process_identification/Process_Identification/Dataset/{}.csv".format(file_name), index=False)
