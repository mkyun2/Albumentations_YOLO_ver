import os
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
dir_path = "C:/MK/GraduateSource/gopro/UGV/" #Img_data Folder
save_path ="aug" #new Folder
save_name ="aug_" #new name ex: 'aug_'+'original name'+'.jpg'


try:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
except OSError:
    print ('not creat directory. ' +  save_path)

for a in os.listdir(dir_path):
    if a.endswith(".jpg"): 
    
        bbox=[]
        category_ids = []
        category_ids_to_name = {0:'tracker', 1:'ugv'} # define class id and name
        fileName = a.split(".")[0]
        image = plt.imread(dir_path+fileName+'.jpg') 
        try:
            fin = open(dir_path+fileName+'.txt',"rt")

            image_size = image.shape
            print(image.shape)
            lines= fin.readlines()
            fin.close()
            fout = open(save_path+'/'+save_name+fileName+'.txt',"wt")
            for line in lines:
                # index 0: class index 1: X index 2: Y index 3: endX index 4: endY
                clsindex = int(line.split(" ")[0])
                x = float(line.split(" ")[1].split(" ")[0])
                y = float(line.split(" ")[2].split(" ")[0])
                width = float(line.split(" ")[3].split(" ")[0])
                height = float(line.split(" ")[4].split("\n")[0])
                bbox.append([x, y, width, height])
                category_ids.append(clsindex)    
                    
            
            

            transform = A.Compose([
                A.RandomBrightnessContrast(),
                A.Transpose(),
                A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=20),
                #A.RandomCrop(1024,1024),
                A.HorizontalFlip(p=0.7),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

            transformed = transform(image=image, bboxes=bbox, category_ids=category_ids)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            img = transformed_image.copy()
            image_size = transformed_image.shape
            #plt.imshow(transformed_image)
            i=0
            for transformed_bbox in transformed_bboxes:
                
                #print(transformed_bbox)
                
                cls_i = category_ids[i]
                x, y, w, h= transformed_bbox
                #print('class: ',category_ids_to_name[cls_i])
                #print('og: ',x,y,w,h)
                xmin = (x-w/2)*image_size[1]
                ymin = (y-h/2)*image_size[0]
                xmax = (x+w/2)*image_size[1]
                ymax = (y+h/2)*image_size[0]
                #print('after',xmin,ymin,xmax,ymax)
                fout.write(str(cls_i)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n')
                rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='red', fill=False)
                ax = plt.gca()
                ax.add_patch(rect)
                i=i+1
            saveimg = cv2.cvtColor(transformed_image,cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path+'/'+save_name+a,saveimg)

            print('Augmented',dir_path,fileName)
        except IOError:
            print('not exists file',dir_path,fileName,'.txt')
            continue
        fout.close()
    
        #plt.xticks([]); plt.yticks([])
        #plt.show()
