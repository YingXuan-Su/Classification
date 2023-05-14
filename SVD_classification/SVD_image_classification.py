'''
Reference : https://web.cs.ucdavis.edu/~bai/ECS130/References/EldenChap10.pdf
'''
import h5py 
import numpy as np
import cv2
import pandas as pd

def main():

    train_features, train_labels, test_features, test_labels = getUspDataSet(r'Mean_classification\usps.h5')
    print(f'There are {train_features.shape[0]} images in train dataset.')
    print(f'There are {test_features.shape[0]} images in test dataset.')

    train_data = {}
    for i in range(len(list(set(train_labels)))):
        idx = np.argwhere(train_labels==i)
        train_data[i] = train_features[idx].squeeze().transpose()  # (number of images, 1,256)-> (1,256) -> (256,1)
        # print(f'There are {train_data[i].shape[1]} images in class number {i}.')

    # show_one_image(train_data[3][:,0].reshape(16,16))  # make sure the image.

    # Perform SVD on all images of all categories in the training dataset to get the matrix U (left singular matrix).
    U_list = []
    for i in range(len(train_data)):
        U, _, _ = np.linalg.svd(train_data[i])
        U_list.append(U)

    index = list(range(10))
    index.append('Total Acc')
    df = pd.DataFrame({}, index=index)

    for k in range(1, 21):   # run all k (the number of eigenvalue used).

        # predict all images in test dataset.
        pred_list = []
        for i in range(len(test_features)):
            pred_list.append( predict(U_list, test_features[i], k) )
    
        # calculate the accurancy of all categories(number 0~9).
        acc_list = []
        for i in range(10):
            idx = np.argwhere(test_labels==i)
            num_correct = sum([test_labels[i[0]]==pred_list[i[0]] for i in idx])
            acc_list.append(num_correct/len(idx))

        # calculate the accurancy of the entire test dataset.
        num_correct = sum([i==j for (i,j) in zip(test_labels, pred_list)])
        print('k={}, Accuracy = {:.3f}%'.format(k, num_correct/len(test_labels)*100))
        acc_list.append(num_correct/len(test_labels))
        
        df[k] = acc_list
    
    # Obtain the value of k for which the accuracy is highest.
    best_k = df.idxmax(axis=1)['Total Acc']
    print(f'the best k is {best_k}')

    fig = df[best_k].plot.line()
    fig.figure.savefig('SVD_classification\SVD_performance.png')

    # convert the numbers in a dataframe to percentages and specify the number of decimal.
    df = df.applymap(lambda x: ("{:.3f}%".format(x * 100)))
    df.to_csv('SVD_classification\SVD_performance.csv')

    

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