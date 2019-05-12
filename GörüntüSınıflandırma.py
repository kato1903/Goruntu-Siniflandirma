from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
from sklearn.model_selection import KFold, StratifiedKFold

from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# Eğitim için kullanılacak verinin dosya yolu
train_path = "train"
# train klasörü içinde her bir sınıf için bir dosya olacak ve
#içindeki örnekler 1,2,3 .. olarak isimlendirilmeli


# Random Forest için kullanılacak ağaç sayısı
num_trees = 100

# Test verisinin oranı
test_size = 0.20

# Her bir sınıfdaki örnek sayısı
class_count = 20

# Makine öğrenmesi algoritmaları için özellik oluşturucular

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Etiketleri Alma
train_labels = os.listdir(train_path)

# Özellik Vektörlerini ve Etiketleri tutmak için değişkenler
global_features = []
labels = []

i, j = 0, 0
k = 0

# Eğitim verileri ve sınıfları okumak için döngü
for training_name in train_labels:
    
    dir = os.path.join(train_path, training_name)
    dir = train_path + "/" + training_name
    
    current_label = training_name

    k = 1
    
    # Her bir sınıf için okuma
    for x in range(1,class_count+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"
        print(file)
             
        image = cv2.imread(file)
        
        #image = cv2.resize(image, fixed_size)
        
        # Özelliklerin Çıkarımı
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print ("{} Türünün Okunması Tamamlandı".format(current_label))
    j += 1


targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)


# Özellik Vektörünün 0 ile 1 arasına normalize edilmesi
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

# Makine Öğrenmesi Algoritmaları
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees)))
models.append(('NB', GaussianNB()))

results = []
names = []
scoring = "accuracy"

global_features = rescaled_features
global_labels = target

# Verilerin Eğitim ve Test Olarak ikiye ayrılması
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          )

import warnings
warnings.filterwarnings('ignore')

# 5-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=5, random_state=7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Algoritma Karşılaştırmaları için grafikler
fig = pyplot.figure()
fig.suptitle('Algoritmaların Karşılaştırması')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

names = ['LR', 'KNN',"DTC","RF","NB"]
values = [1, 10, 100,1,2,3,4]
values2 = []

names2 = ["100x100","300x300","480x480","800x600","1000x1000"]

for i in results:
    values2.append(i.mean())

fig = pyplot.figure()
plt.title('Algoritmaların Karşılaştırması')
plt.ylabel('Ortalama Doğruluk Değerleri')
ax = fig.add_subplot(111)

plt.bar(names, values2, align='center', alpha=0.5)
ax.set_xticklabels(names)
fig.set_size_inches(10, 8)
fig.savefig('Results.png', dpi=100)
pyplot.show()

fig = pyplot.figure()
plt.title('Algoritmaların Karşılaştırması')
plt.ylabel('Box Plot')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
fig.set_size_inches(10, 8)
fig.savefig('Results2.png', dpi=100)
pyplot.show()
