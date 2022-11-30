import lost_ds as lds
import os

from yolov3.configs import *
from yolov3.yolov4 import read_class_names

def write_data_txt(path, ds):
    with open(path, "w") as file:           
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

# convert from xcycwh rel to x1y1x2y2 abs
ds.df = lds.transform_bbox_style('x1y1x2y2', ds.df)
ds.df = lds.to_abs(ds.df)

# validation set is not needed
train, test, val = ds.split_by_img_path(0.2, 0.0)

# get classes
train_classes_dict = read_class_names(TRAIN_CLASSES)

# build train txt
write_data_txt(TRAIN_ANNOT_PATH, train)        
# build test txt
write_data_txt(TEST_ANNOT_PATH, test)



