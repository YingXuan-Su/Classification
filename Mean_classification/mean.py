import h5py 
from functools import reduce
import numpy as np
import cv2

def main():
    train_features, train_labels, test_features, test_labels = getUspDataSet(r'Mean_classification\usps.h5')
    print(f'train_feature 有 {train_features.shape[0]} 張')
    print(f'test_feature 有 {test_features.shape[0]} 張')

    # reshape to (16,16)
    train_features = np.array([np.array(train_features[i]).reshape(16,16) for i in range(len(train_features))])
    test_features = np.array([np.array(test_features[i]).reshape(16,16) for i in range(len(test_features))])
    print(f'每一張的shape為{train_features[0].shape}')

    label_type = list(set(train_labels))

    # mean_train = train_features.mean(0)
    # show_one_img(mean_train)

    means = get_mean(features=train_features, labels=train_labels)
    
    show_one_img(means.reshape((-1, 16)))
    show_one_img(means[0])
    # print(pred_by_mean(means, test_features[0]))
    # show_one_img(test_features[0])

    pred_list = []
    for i in range(len(test_features)):
        pred_list.append(pred_by_mean(means, test_features[i]))

    acc_list = []
    for i in range(10):
        idx = np.argwhere(test_labels==i)
        num_correct = sum([test_labels[i[0]]==pred_list[i[0]] for i in idx])
        acc_list.append(num_correct/len(idx))

    num_correct = sum([i==j for (i,j) in zip(test_labels, pred_list)])
    print(f'Accuracy = {num_correct/len(test_labels)}')
    acc_list.append(num_correct/len(test_labels))
    
    for i in range(len(acc_list)-1):
        print(f'Acc of Number {i} : {acc_list[i]}')
    print(f'Total Acc : {acc_list[-1]}')

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

# mean method
def get_mean(features, labels):
    label_type = list(set(labels))
    mean_imgs = np.zeros((len(label_type), features.shape[1], features.shape[2]))
    for i in range(len(label_type)):
        idx = np.argwhere(labels==i)
        mean_imgs[i] = features[idx].mean(0)
    return mean_imgs

def two_norm(a, b):
    return np.inner(a, b).sum()

def pred_by_mean(means, pred):
    two_morms = []
    for i in range(means.shape[0]):
        two_morms.append( np.linalg.norm(means[i] - pred, 2 ) )
    return np.array(two_morms).argmin()

if __name__ == '__main__':
    main()


