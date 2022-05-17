import os
import shutil

os.rename("./tiny-imagenet-200/val", "./tiny-imagenet-200/originval")
os.mkdir("./tiny-imagenet-200/val")
for filename in os.listdir("./tiny-imagenet-200/train"):
    if filename in os.listdir("./tiny-imagenet-200/originval"):
        pass
    else:
        os.mkdir(os.path.join("./tiny-imagenet-200/val", filename))
with open("./tiny-imagenet-200/originval/val_annotations.txt") as f:
    labels = f.readlines()
for i, label in enumerate(labels):
    filename = label.split("\t")[1]
    src = "./tiny-imagenet-200/originval/images/val_" + str(i) + ".JPEG"
    dst = "./tiny-imagenet-200/val/" + filename + "/val_" + str(i) +".JPEG"
    shutil.copyfile(src, dst)