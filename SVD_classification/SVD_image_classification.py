'''
Reference : https://web.cs.ucdavis.edu/~bai/ECS130/References/EldenChap10.pdf
'''
import h5py 
from functools import reduce
import numpy as np
import cv2
import pandas as pd

def main():

    train_features, train_labels, test_features, test_labels = getUspDataSet(r'Mean_classification\usps.h5')
    print(f'Train Dataset 有 {train_features.shape[0]} 張')
    print(f'Test Dataset 有 {test_features.shape[0]} 張')

    #train_features = np.array([train_features[i].reshape(16,16) for i in range(train_features.shape[0])])
    #train_features = np.array([test_features[i].reshape(16,16) for i in range(test_features.shape[0])])

    train_data = {}
    for i in range(len(list(set(train_labels)))):
        idx = np.argwhere(train_labels==i)
        train_data[i] = train_features[idx].squeeze().transpose()
        #print(f'{i}有{train_data[i].shape[1]}張')

    #show_one_image(train_data[1][:,0].reshape(16,16))  # make sure the image

    U_list = []
    for i in range(len(train_data)):
        U, _, _ = np.linalg.svd(train_data[i])
        U_list.append(U)

    index=list(range(10))
    index.append('Total Acc')
    acc_df = pd.DataFrame({}, index=index)
    for k in range(1, 21):   
        pred_list = []
        for i in range(len(test_features)):
            pred_list.append( predict(U_list, test_features[i], k) )
    
        acc_list = []
        for i in range(10):
            idx = np.argwhere(test_labels==i)
            num_correct = sum([test_labels[i[0]]==pred_list[i[0]] for i in idx])
            acc_list.append(num_correct/len(idx))

        num_correct = sum([i==j for (i,j) in zip(test_labels, pred_list)])
        print(f'{k} : {num_correct}/{len(test_labels)}')
        acc_list.append(num_correct/len(test_labels))
        
        acc_df[k] = np.around(acc_list, decimals=3)
    
    best_k = acc_df.idxmax(axis=1)['Total Acc']
    print(f'the best k is {best_k}')
    fig = acc_df[best_k].plot.line()
    fig.figure.savefig('performance.png')
    acc_df.to_csv('SVD_performance.csv')

    

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

def show_one_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict(U_list, test_feature, k):

    rel_residual_list = []
    for i in range(10):  # run all number (0~9)
        I = np.diag([1]*256)
        U_k = U_list[i][:,:k]
        residual = np.linalg.norm( np.dot((I - np.dot(U_k, U_k.transpose()) ), test_feature), 2)
        relative_residual = residual / np.linalg.norm(test_feature, 2)
        rel_residual_list.append(relative_residual)

    return np.argmin(rel_residual_list)


if __name__ == '__main__':
    main()