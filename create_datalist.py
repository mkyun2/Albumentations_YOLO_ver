import os

dir_path = "C:/MK/GraduateSource/aug/"
f = open('trainAug.txt','w')

for a in os.listdir(dir_path):
    if a.endswith(".jpg"):
        fileName = a.split(".")[0]
        print(fileName)
        f.write("/home/biorobotics/mkyun/gopro_train/aug/"+fileName+".jpg\n")
    if a.endswith(".png"):
        fileName = a.split(".")[0]
        print(fileName)
        f.write("/home/biorobotics/mkyun/gopro_train/UGV/"+fileName+".png\n")    
f.close()
