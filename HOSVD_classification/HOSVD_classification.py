'''
Reference : 
https://epubs.siam.org/doi/pdf/10.1137/S0895479896305696
https://zhuanlan.zhihu.com/p/381312642
https://blog.csdn.net/weixin_44737922/article/details/125250976
'''

import h5py 
from functools import reduce
import numpy as np
import cv2
import tensorly as tl
from tqdm import tqdm

def main():

    NUM_CLASS = 10
    train_features, train_labels, test_features, test_labels = getUspDataSet(r'HOSVD_classification\usps.h5')
    print(f'train_feature 有 {train_features.shape[0]} 張')
    print(f'test_feature 有 {test_features.shape[0]} 張')

    # reshape to (16,16)
    train_features = np.array([np.array(train_features[i]).reshape(16,16) for i in range(len(train_features))])
    test_features = np.array([np.array(test_features[i]).reshape(16,16) for i in range(len(test_features))])
    print(f'每一張的shape為{train_features[0].shape}')

    k=12

    Ss, U_1s, U_2s = {}, {}, {}
    for i in tqdm(range(NUM_CLASS)):
        # Step 1 : arrange the dataset for all classes respectively
        idx = np.argwhere(train_labels==i)
        train_data = train_features[idx].squeeze()  # (1194, 16, 16)
        train_data = np.transpose(train_data, (1,2,0))  # (16, 16, 1194)

        # show_one_image(train_data[:,:,0])
        S, U_1, U_2 = HOSVD(train_data)  
        Ss[i], U_1s[i], U_2s[i] = S, U_1, U_2

    pred_list = []
    for i in tqdm(range(test_features.shape[0])):
        k = 12
        pred_list.append(predict(Ss, U_1s, U_2s, test_features[i], k))
    num_correct = sum([i==j for (i,j) in zip(test_labels, pred_list)])
    print(f"Acc={num_correct/len(test_labels)}")
    
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

# Do mode-1~3 Matricization and SVD to get U^(n), S
def HOSVD(data):
    U_1, _, _ = np.linalg.svd(tl.unfold(data, 0))
    U_2, _, _ = np.linalg.svd(tl.unfold(data, 1))
    U_3, _, _ = np.linalg.svd(tl.unfold(data, 2))
    #print(U_1.shape, U_2.shape, U_3.shape)  # (16, 16) (16, 16) (1194, 1194)
    S = tl.tenalg.multi_mode_dot(data, [U_1, U_2, U_3], modes=[0,1,2], transpose=True)
    #print(S.shape)  # (16,16, numer of pictures) same as data
    return S, U_1, U_2

# Predict
def predict(Ss, U_1s, U_2s, test_feature, k):
    z_list = []
    residual_list = []
    for i in range(len(Ss)):  # run all classes
        S, U_1,  U_2 = Ss[i], U_1s[i], U_2s[i]
        sum = np.zeros(test_feature.shape)
        for j in range(k):
            A_j = tl.tenalg.multi_mode_dot(S[:,:,j], [U_1, U_2], modes=[0,1])  #(16,16)
            z_j = np.tensordot(test_feature, A_j) / np.tensordot(A_j, A_j)  # is a scale
            sum = sum + z_j * A_j
        #show_one_image(sum)
        residual = np.linalg.norm(test_feature - sum, 'fro')  # F-noem
        residual_list.append(residual)

    return np.argmin(residual_list)


if __name__ == '__main__':
    main()