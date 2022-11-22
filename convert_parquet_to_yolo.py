import lost_ds as lds
import os
import json

from yolov3.configs import *


Dataset_names_path = "./model_data/marvel_names.txt"

label_list = []

train_classes_dict = {}

def write_data_txt(path, ds):
    with open(path, "w") as file:           
        for imgpath in ds.groupby('img_path'):
            line = ''
            line += imgpath[0]
            
            for coor, label, id in zip(imgpath[1]['anno_data'], imgpath[1]['anno_lbl'],  imgpath[1]['anno_lbl_id']):
                line += f' {int(coor[0])},{int(coor[1])},{int(coor[2])},{int(coor[3])},'
                line += f'{int(id)}'
                
                train_classes_dict[int(id)] = label[0]
                
            file.write(line+'\n')


    with open('data.json', 'w') as fp:
        json.dump(train_classes_dict, fp)

init_ds = lds.LOSTDataset(ANNO_DATA_PATH)

ds = lds.LOSTDataset(lds.remap_img_path(init_ds.df, 
                                        new_root_path=IMG_PATH, 
                                        col='img_path'))
    
ds.df = ds.remove_empty()

# convert from xcycwh rel to x1y1x2y2 abs
ds.df = lds.transform_bbox_style('x1y1x2y2', ds.df)
ds.df = lds.to_abs(ds.df)

# label list to get index of labels for train.txt/ test.txt
if os.path.exists(Dataset_names_path):
    with open(Dataset_names_path, 'r') as fp:
        train_classes_dict = json.load(fp)
    
else:
    label_list = list(ds.unique_labels(col='anno_lbl'))
    
    # build label names txt
    with open(Dataset_names_path, "w") as file:           
        for label in label_list:

            file.write(str(label)+'\n')
    
# val is not needed
train, test, val = ds.split_by_img_path(0.2, 0.0)


# build train txt
write_data_txt(TRAIN_ANNOT_PATH, train)        
# build test txt
write_data_txt(TEST_ANNOT_PATH, test)



