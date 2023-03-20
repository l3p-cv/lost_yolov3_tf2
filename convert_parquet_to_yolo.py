import lost_ds as lds
import os
import json
import numpy as np
import pandas as pd

from yolov3.configs import *
from yolov3.yolov4 import read_class_names

def write_data_txt(path, ds):
    with open(path, "a") as file:           
        for imgpath in ds.groupby('img_path'):
            line = ''
            line += imgpath[0]
            
            for coor, label in zip(imgpath[1]['anno_data'], 
                                   imgpath[1]['anno_lbl']):
                
                line += f' {int(coor[0])},{int(coor[1])},{int(coor[2])},{int(coor[3])},'
                key_index = list(train_classes_dict.values()).index(label[0])
                dict_key = list(train_classes_dict.keys())[key_index]
                line += f'{dict_key}'
                
                
            file.write(line+'\n')
            
# calculate bbounding box from points
def minmax(a):
    x=[]
    y=[]
    for i,j in a:
        x.append(i)
        y.append(j)
    return np.array([min(x), min(y), max(x), max(y)])

def line_to_bbox(ds, loop):
    bbox_df = ds.df[ds.df['anno_dtype'] == 'bbox']
    line_df = ds.df[ds.df['anno_dtype'] == 'line']
    line_df.reset_index(drop=True, inplace=True)


    for i, xy in enumerate(line_df['anno_data']):
        line_df.at[i,'anno_data'] = minmax(xy)
        line_df.at[i,'anno_style'] = 'x1y1x2y2'
        line_df.at[i,'anno_dtype'] = 'bbox'

        ds.df = pd.concat([bbox_df, line_df])
        ds.transform_bbox_style('xcycwh', inplace=True)

        file = f'LOST_Annotation_{loop}.parquet'
        ds.to_parquet(os.path.join(root, file))
    return ds
    
# build list of parquets 
anno_data = []

for root, _, files in os.walk(os.path.abspath(TRAIN_ANNO_DATA_PATH)):
    for file in files:
        if file.endswith(('.parquet')):
            anno_data.append(os.path.join(root, file))
            
        
init_ds = lds.LOSTDataset(anno_data)

ds = lds.LOSTDataset(lds.remap_img_path(init_ds.df, 
                                        new_root_path=TRAIN_IMG_PATH, 
                                        col='img_path'))
    
ds.remove_empty(inplace=True)
ds.df = ds.df[ds.df['anno_lbl'] != '']

# convert from xcycwh rel to x1y1x2y2 abs
ds.df = lds.transform_bbox_style('x1y1x2y2', ds.df)
ds.df = lds.to_abs(ds.df)

# for semitautomatic pipeline data
if TRAIN_IN_LOOPS:
    if not os.path.isfile(TRAIN_ITER_FILE):
        loop = 0
        with open(TRAIN_ITER_FILE, 'w') as fp:
                
            iter_dict = {'iter': loop}
            json.dump(iter_dict, fp)
    else:
        with open(TRAIN_ITER_FILE, 'r') as fp:
            iter_dict = json.load(fp)
            loop = iter_dict['iter'] + 1    
            iter_dict = {'iter': loop}
        with open(TRAIN_ITER_FILE, 'w') as fp:
            json.dump(iter_dict, fp)
    
    ds.df = ds.df[ds.df['img_iteration']==loop]
    ds_bbox = line_to_bbox(ds, loop)
    
    # validation set is not needed
    train, test, val = ds_bbox.split_by_img_path(0.3, 0.0)

# for pipeline data without iterations
else:    
    ds_bbox = line_to_bbox(ds, loop='')
    
    # validation set is not needed
    train, test, val = ds_bbox.split_by_img_path(0.2, 0.0)

# get classes
train_classes_dict = read_class_names(TRAIN_CLASSES)

# build train txt
write_data_txt(TRAIN_ANNOT_PATH, train)        
# build test txt
write_data_txt(TEST_ANNOT_PATH, test)



