from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from skimage.io import imread
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_preprocessing import preprocessing_data, label_to_integer, integer_to_label
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
import cv2 as cv2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
from sklearn.utils import shuffle
import random

def major_color_hsv(image_array):
    if image_array is None:
        raise ValueError("L'image est vide ou n'a pas pu être chargée.")

    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    blue_count = cv2.countNonZero(mask_blue)
    red_count = cv2.countNonZero(mask_red)

    if blue_count > red_count:
        return 1.
    else:
        return 0.
    
def pca_color_features(h, s, k=2):
    data = np.column_stack((h, s))
    pca = PCA(n_components=k)
    pca_data = pca.fit_transform(data)
    return pca_data

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.6),
    A.MedianBlur(blur_limit=7, p=0.5),
    A.RandomBrightnessContrast(p=0.4),

    ToTensorV2()
])

imgs_names, imgs_bb, classes_indices=  preprocessing_data()
signs = []
hog_image = [] 
fd = []
labels = []
hist_bins = 8
hist_ranges = [0, 180] # Calcul sur les canaux de teinte et de saturation de l'espace colorimétrique HSV
hist_size = [hist_bins, hist_bins]

r_hog = []
g_hog=[]
b_hog=[]
n=4
img_name = []
for i in range(len(imgs_names)):
    img = imread(imgs_names[i])
    for bb in imgs_bb[i]:
        if not bb[4].startswith('f'):
            labels.append(bb[4])
            signs.append(img[bb[1]:bb[3],bb[0]:bb[2],:])
            img_name.append(imgs_names[i])
augmented_signs_tensors = []
augmented_labels = []
all_img_name = img_name.copy()

for i in range(2):
    all_img_name+= img_name
    for image, label in zip(signs, labels):
        # Appliquer les transformations
        try :
            augmented = transform(image=image, labels=label)
            augmented_signs_tensors.append(augmented['image'])
            augmented_labels.append(augmented['labels'])
        except:
            print("Erreur augmentation")

augmented_signs = [tensor.numpy() for tensor in augmented_signs_tensors]
augmented_signs = [np.transpose(array, (1, 2, 0)) for array in augmented_signs]
signs = signs + augmented_signs        
labels = labels +augmented_labels 

j=0       
for sign in signs:      
    try :
        sign = cv2.resize(sign, (40,40))
        fd_k, hog_k = hog(sign, orientations=8, pixels_per_cell=(5,5), cells_per_block=(2, 2), visualize=True, multichannel=True)
        color_feature = major_color_hsv(sign) * np.ones(200)
        fd.append(np.concatenate((color_feature, fd_k)))
    except :
        print(f"error hog {bb[4]}")
        labels.pop(j)
        signs.pop(j)
        j-=1
    j+=1



hog_features = np.array(fd)
labels = np.array(labels).reshape(-1, 1)
data_frame = np.hstack((hog_features, labels))
data_frame, signs, img_name = shuffle(data_frame, signs, all_img_name)
percentage = 80
partition = int(len(hog_features)*percentage/100)
x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()
print("fit")

clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))    
joblib.dump(clf, 'svm_model.joblib') 


false_predictions = np.where(y_pred != y_test)[0]
false_predictions = random.sample(list(false_predictions), 10)

fig, axs = plt.subplots(2, 5, figsize=(15, 6))

for i, idx in enumerate(false_predictions):
    
    row, col = i // 5, i % 5
    axs[row, col].imshow(signs[idx+len(y_train)])
    axs[row, col].set_title(f"Pred: {y_pred[idx]}, True: {y_test[idx]} {all_img_name[idx+len(y_train)]}")
    axs[row, col].axis('off')

plt.tight_layout()
plt.show()
