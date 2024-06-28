import pandas as pd
import shutil
import os

def klasorleriOlustur(hastalik):
        try:
            os.makedirs("train")
        except FileExistsError:
            pass
        try:
            os.makedirs("test")
        except FileExistsError:
            pass
        try:
            os.makedirs("validation")
        except FileExistsError:
            pass
        try:
            os.makedirs("train/" + hastalik)
        except FileExistsError:
            pass
        try:
            os.makedirs("test/" + hastalik)
        except FileExistsError:
            pass
        try:
            os.makedirs("validation/" + hastalik)
        except FileExistsError:
            pass
        
data = pd.read_csv("gt.csv", usecols = ['image','MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK'])
images = data["image"].tolist()
mel = data["MEL"].tolist()
nv = data["NV"].tolist()
bcc = data["BCC"].tolist()
ak = data["AK"].tolist()
bkl = data["BKL"].tolist()
df = data["DF"].tolist()
vasc = data["VASC"].tolist()
scc = data["SCC"].tolist()
unk = data["UNK"].tolist()

"""
print(int(sum(nv))) #12875
print(int(sum(mel))) #4522
print(int(sum(bcc))) #3323
print(int(sum(bkl))) #2624
print(int(sum(ak))) #867
print(int(sum(df))) #239
print(int(sum(vasc))) #253
print(int(sum(scc))) #628
print(int(sum(unk))) #0

"""

imageList = list()
idx = 0
for i in bkl:
    if i == 1:
        imageList.append(images[idx] + ".jpg")
        if len(imageList) == 2000:
            break
    idx += 1
klasorleriOlustur("BKL")     
klasor = "train/BKL"
sayac = 0
for i in imageList:
    if sayac == 1000:
        klasor = "test/BKL"
    elif sayac == 1500:
        klasor = "validation/BKL"
    shutil.copy(('images/'+ i), klasor)
    sayac += 1