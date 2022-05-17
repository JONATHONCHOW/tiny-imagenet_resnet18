import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_filename(index):
    folder_index = index // 50
    image_index = index % 50
    folder = DIR_LIST[folder_index]
    image = os.listdir(os.path.join("./tiny-imagenet-200/val", folder))[image_index]
    return (folder, image)

DIR_LIST = os.listdir("./tiny-imagenet-200/train")
index_word = ["" for i in range(200)]
with open("./tiny-imagenet-200/words.txt") as f:
    labels = f.readlines()
for j in range(0, 200):
    for label in labels:
        label = label.split("\t")
        if label[0] == DIR_LIST[j]:
            index_word[j] = label[1][:-1]

# gpu to cpu
pred1 = torch.load("epoch10.dat").cpu().numpy()
pred2 = torch.load("epoch20.dat").cpu().numpy()

n = 10
index = np.sort(np.random.choice(np.where(pred1[0] != pred2[0])[0], n, replace=False))
files = ["" for i in range(n)]
for i in range(n):
    folder, image = get_filename(index[i])
    files[i] = os.path.join("./tiny-imagenet-200/val", folder, image)
for i in range(n):
    figure = mpimg.imread(files[i])
    plt.imshow(figure)
    plt.axis("off")
    plt.show()
    print("answer:", index_word[index[i] // 50])
    print("epoch10:", [index_word[j] for j in pred1[:,index[i]]])
    print("epoch20:", [index_word[j] for j in pred2[:,index[i]]])