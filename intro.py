import h5py 
from functools import reduce
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
def main():
    
    # acc = [0.8067 ,0.8794 ,0.8979 ,0.9083 ,0.9193 ,0.9203 ,0.9268 ,0.9327 ,0.9332 ,0.9347 
    #        ,0.9362 ,0.9452 ,0.9442 ,0.9442 ,0.9442 ,0.9447 ,0.9422 ,0.9407 ,0.9392 ,0.9412 ]

    # plt.plot(range(1,21), acc)
    # plt.grid(linestyle = '-')
    # plt.xticks(range(1,21),range(1,21))
    # plt.title('Total Accurancy for different k.')
    # plt.xlabel('k')
    # plt.ylabel('Total Accurancy')
    # plt.show()

    train_features, train_labels, test_features, test_labels = getUspDataSet(r'Mean_classification\usps.h5')
    print(f'There are {train_features.shape[0]} images in train dataset.')
    print(f'There are {test_features.shape[0]} images in test dataset.')
    train_features = np.array([np.array(train_features[i]).reshape(16,16) for i in range(len(train_features))])
    test_features = np.array([np.array(test_features[i]).reshape(16,16) for i in range(len(test_features))])


    img0, img1 = train_features[0,:,:], train_features[5,:,:]
    for i in range(1, 5):  
        img0 = np.concatenate((img0, train_features[i,:,:]), axis=1)
        img1 = np.concatenate((img1, train_features[i+5,:,:]), axis=1)
    img = np.concatenate((img0, img1), axis=0)
    
    cv2.imwrite('hand_write_image.png', 255*img) 

# https://www.kaggle.com/datasets/bistaumanga/usps-dataset
def getUspDataSet(path):
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        x_Train = train.get("data")[:]
        y_Train = train.get("target")[:]
        test = hf.get('test')
        x_Test = test.get("data")[:]
        y_Test = test.get("target")[:]
    return x_Train, y_Train, x_Test, y_Test

def show_one_img(data):
    cv2.imshow('img', data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
