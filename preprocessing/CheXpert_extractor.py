import os
import pickle
import numpy as np
import pandas as pd
import cv2

PATH_CHEXPERT = '../data/CheXpert-v1.0-small/train.csv'


patients = os.listdir(r'C:\Users\Rufina\Desktop\repos\MedicalReports_v_1.0\data\CheXpert-v1.0-small\train')[:20_000]
N = 12
IMAGES , TAGS= [], []

train_set = pd.read_csv(PATH_CHEXPERT)
train_set.drop(columns=['Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 'No Finding', 'Support Devices'], inplace=True)

keys = {'nan':0, 0:1, -1:1, 1:1}
k = 0
for row in train_set.values:
    tag = np.zeros(N)
    for i in range(len(row[1:])):
        try: tag[i] = keys[row[1:][i]]
        except KeyError: tag[i] = 0
    patient_numb = row[0][26:38]
    if patient_numb in patients:
        # img = cv2.imread(f'../data/{row[0]}')
        # img = cv2.resize(img, (224,224))
        new_img_path = f'{patient_numb}_{row[0][39:45]}_{row[0][46:]}'
        # cv2.imwrite(f'../data/CheXpert-v1.0-small/train_small/{new_img_path}', img)
        IMAGES.append(f'train_small/{new_img_path}')
        TAGS.append(tag)

data = {'img_path': IMAGES, 'tag': TAGS}
CHEXPERT_IMG_TAG = pd.DataFrame.from_dict(data)

print(CHEXPERT_IMG_TAG.shape)

with open('CHEXPERT_IMG_TAG.pickle', 'wb') as f:
    pickle.dump(CHEXPERT_IMG_TAG, f)

