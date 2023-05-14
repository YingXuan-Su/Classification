import h5py 
import numpy as np
import cv2
import pandas as pd

from sklearn.metrics import precision_score,recall_score,f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def main():
    
    train_features, train_labels, test_features, test_labels = getUspDataSet(r'sklearn_models\usps.h5')
    print(f'There are {train_features.shape[0]} images in train dataset.')
    print(f'There are {test_features.shape[0]} images in test dataset.')

    Classifiers=[
        ["Random Forest", RandomForestClassifier()],
        ["Support Vector Machine", SVC()],
        ["KNN", KNeighborsClassifier()],
        ["Decision Tree", DecisionTreeClassifier()],
        ["Naive Bayes", GaussianNB()]
    ]

    name_list = []
    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    
    index = list(range(10))
    index.append('Total Acc')
    df_acc = pd.DataFrame({}, index=index)

    for name, classifier in Classifiers:

        classifier = classifier
        classifier.fit(train_features, train_labels)
        pred = classifier.predict(test_features) 

        name_list.append(name)
        acc_list.append(accuracy_score(test_labels, pred))
        recall_list.append(recall_score(test_labels, pred, average='micro'))  # , average='micro'
        precision_list.append(precision_score(test_labels, pred, average='micro'))
        f1_list.append(f1_score(test_labels, pred, average='micro'))

        acc_per_categories = []
        for i in range(10):
            idx = np.argwhere(test_labels==i)
            num_correct = sum([test_labels[i[0]]==pred[i[0]] for i in idx])
            acc_per_categories.append(num_correct/len(idx))
        acc_per_categories.append(acc_list[-1])
        df_acc[name] = acc_per_categories

        print('model : {}, Accuracy = {:.3f}%'.format(name, acc_list[-1]*100))

    df = pd.DataFrame({"name":name_list, "accuracy":acc_list, "recall":recall_list, "precision":precision_list, "f1_score":f1_list, "accuracy":acc_list })
    # convert the numbers in a dataframe to percentages and specify the number of decimal.
    for i in range(1, len(df.columns)):
        df[df.columns[i]] = df[df.columns[i]].apply(lambda x: "{:.3}%".format(x*100))
    df.to_csv('sklearn_models/performance.csv')
    df_acc.to_csv('sklearn_models/Acc.csv')


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

if __name__ == '__main__':
    main()