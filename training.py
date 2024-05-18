# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:30:40 2024

@author: louis
"""


from skimage.io import imread
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_preprocessing import preprocessing_data
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

imgs_names, imgs_bb, classes_indices=  preprocessing_data()
n = sum (len(classes) for classes in imgs_bb)+1
signs = []
hog_image = [] 
fd = []
labels = []
j=0
for i in range(len(imgs_names)):
    img = imread(imgs_names[i])
    for bb in imgs_bb[i]:
        labels.append(bb[4])
        signs.append(img[bb[1]:bb[3],bb[0]:bb[2],:])
        try :
            fd_k, hog_k = hog(signs[j], orientations=9, pixels_per_cell=(int(signs[j].shape[0])/4, 
            int(signs[j].shape[1])/4), cells_per_block=(2, 2), visualize=True, multichannel=True)
            fd.append(fd_k)
        except :
            print(f"error hog {bb[4]}")
            j-=1
            labels.pop()
            signs.pop()
        j+=1

labels_df = pd.DataFrame(labels)
labels_df.to_csv('labels.csv', index=False, header=False)


clf = svm.SVC()
hog_features = np.array(fd)
labels = np.array(labels).reshape(-1, 1)
data_frame = np.hstack((hog_features,labels))
np.random.shuffle(data_frame)
percentage = 80
partition = int(len(hog_features)*percentage/100)
x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()
print("fit")
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

   

# fig, axes = plt.subplots(2, 5)
# for i in range(5):
#     axes[0,i].imshow(signs[i+70])
#     axes[0,i].set_title(f"panneau")
#     axes[0,i].axis("off")

#     axes[1,i].imshow(hog_image[i+70])
#     axes[1,i].set_title(f"hog")
#     axes[1, i].axis("off")