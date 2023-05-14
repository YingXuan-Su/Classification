import h5py 
from functools import reduce
import numpy as np
import cv2
import pandas as pd

def main():
    
    train_features, train_labels, test_features, test_labels = getUspDataSet(r'Mean_classification\usps.h5')
    print(f'There are {train_features.shape[0]} images in train dataset.')
    print(f'There are {test_features.shape[0]} images in test dataset.')

    # reshape to (16,16)
    train_features = np.array([np.array(train_features[i]).reshape(16,16) for i in range(len(train_features))])
    test_features = np.array([np.array(test_features[i]).reshape(16,16) for i in range(len(test_features))])
    print(f'The shape of each images is {train_features[0].shape}.')

    # Calculate the mean of all images of all categories in the training dataset.
    means = get_mean(features=train_features, labels=train_labels)  # (num_class, 16, 16)
    
    # show_one_img(means.reshape((-1, 16)))  # horizontal arrangement.

    # vertical arrangement.
    vert_mean = means[0,:,:]
    for i in range(1, means.shape[0]):  
        vert_mean = np.concatenate((vert_mean, means[i,:,:]), axis=1)
    cv2.imwrite('Mean_classification\mean_image.jpg', 255*vert_mean)  # need to *255
    # show_one_img(vert_mean)  # to chack the mean calculation is true

    # predict all images in test dataset.
    pred_list = []
    for i in range(len(test_features)):
        pred_list.append(pred_by_mean(means, test_features[i]))

    # calculate the accurancy of all categories(number 0~9).
    acc_list = []
    for i in range(10):
        idx = np.argwhere(test_labels==i)
        num_correct = sum([test_labels[i[0]]==pred_list[i[0]] for i in idx])
        acc_list.append(num_correct/len(idx))

    # calculate the accurancy of the entire test dataset.
    num_correct = sum([i==j for (i,j) in zip(test_labels, pred_list)])
    # print('Accuracy = {:.3f}%'.format(num_correct/len(test_labels)*100))
    acc_list.append(num_correct/len(test_labels))
    
    for i in range(len(acc_list)-1):
        print('Acc of Number {} : {:.3f}%'.format(i, acc_list[i]*100 ))
    print('Total Acc : {:.3f}%'.format(acc_list[-1]*100 ))

    idx = list(range(10))
    idx.append('Total Acc')
    df = pd.DataFrame([acc_list], columns = idx, index=['Accurancy'])

    # convert the numbers in a dataframe to percentages and specify the number of decimal.
    df = df.applymap(lambda x: ("{:.3f}%".format(x * 100)))

    df.to_csv('Mean_classification\mean_performance.csv')


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


