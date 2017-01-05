import os
import numpy as np
import scipy.io as sio
import cv2
import preprocess

def create_train_dataset():
    return create_dataset(['train/'], 'digitStruct.mat')

def create_extra_dataset():
    return create_dataset(['extra/'], 'digitStruct.mat')

def create_test_dataset():
    return create_dataset(['test/'], 'digitStruct.mat')

def create_dataset(dlist, fname = 'digitStruct.mat'):
    x = []
    y = []

    for directory in dlist:
        matrix = sio.loadmat(directory + fname)
        mr, mc = matrix['digitStruct'].shape
        length = mr * mc
        matrix = matrix['digitStruct'].reshape([length])
        edge, small_patch = 0, 0

        for i in range(0, length):
            img_fname = matrix[i]['name'][0]
            img = cv2.imread(directory + img_fname)
            br, bc = matrix[i]['bbox'].shape
            bbox = matrix[i]['bbox'].reshape([br * bc])

            for element in bbox:
                label = int(element['label'][0][0])
                top = int(element['top'][0][0])
                height = int(element['height'][0][0])
                left = int(element['left'][0][0])
                width = int(element['width'][0][0])

                if(top < 2 or left < 2):
                    #print("EDGE")
                    edge += 1
                elif(width < 2 or height < 2):
                    #print("SMALL PATCH")
                    small_patch += 1
                else:
                    sub_img = img[top:(top + height), left:(left + width)]
                    #cv2.imshow('pre' ,sub_img)
                    sub_img = preprocess.pad(sub_img)
                    sub_img = cv2.resize(sub_img, (32, 32))
                    #cv2.destroyAllWindows()
                    #cv2.imshow(directory + img_fname, sub_img)
                    #xj = input()

                    x.append(sub_img)
                    _y = np.zeros([10])
                    _y[label - 1] = 1
                    y.append(_y)

    return (np.array(x), np.array(y))
