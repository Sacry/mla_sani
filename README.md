# Machine Learning Algorithms - Simple and Naive Implementation

As a software engineer with not that strong math background, I had a great difficulty in understanding ML algorithms. It may take me only several minutes to know what an algorithm looks like, but it really takes some time to understand the further details. As a way of learning, I decided to implement some basic ml algorithms from scratch, and this project is the result.

The API is merely a copy from scikit-learn (~~and Keras~~). There’s no parameter check nor any optimization. I tried to focus on the most simple and naive part of these algorithms and keep the code as “dense” as possible. 

Currently implemented algorithms are as follows:

* Supervised
  * [LinearRegression](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/linear_model.py#L38)
  * [Ridge](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/linear_model.py#L110)
  * [Lasso](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/linear_model.py#L119)
  * [ElasticNet](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/linear_model.py#L128)
  * [LogisticRegression](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/linear_model.py#L139)
  * [MLPClassifier](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/neural_network.py#L6)
  * [DecisionTree](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/tree.py)
  * [SVM](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/svm.py)
  * [GaussianNB](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/naive_bayes.py#L5)
  * [KNeighborsClassifier](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/neighbors.py#L5)
  * [BaggingClassifier](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/ensemble.py#L8)
  * [AdaBoostClassifier](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/ensemble.py#L102)
  * [GradientBoostingClassifier](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/ensemble.py#L191)
  * [LinearDiscriminantAnalysis](https://github.com/Sacry/mla_sani/blob/master/mla_sani/supervised/discriminant_analysis.py#L5)
* Unsupervised
  * [KMeans](https://github.com/Sacry/mla_sani/blob/master/mla_sani/unsupervised/cluster.py#L5)
  * [DBSCAN](https://github.com/Sacry/mla_sani/blob/master/mla_sani/unsupervised/cluster.py#L53)
  * [PCA](https://github.com/Sacry/mla_sani/blob/master/mla_sani/unsupervised/decomposition.py)
  * [GaussianMixture](https://github.com/Sacry/mla_sani/blob/master/mla_sani/unsupervised/mixture.py#L6)
* DL Layers
  * [Dense](https://github.com/Sacry/mla_sani/blob/master/mla_sani/nn/layers.py#L64)
  * [Activation](https://github.com/Sacry/mla_sani/blob/master/mla_sani/nn/layers.py#L107)
  * [Conv2D](https://github.com/Sacry/mla_sani/blob/master/mla_sani/nn/layers.py#L138)
  * [MaxPooling2D](https://github.com/Sacry/mla_sani/blob/master/mla_sani/nn/layers.py#L376)
  * [Flatten](https://github.com/Sacry/mla_sani/blob/master/mla_sani/nn/layers.py#L360)
  * [Dropout](https://github.com/Sacry/mla_sani/blob/master/mla_sani/nn/layers.py#L480)

The traditional ML algorithms are implemented in the way that it can be used just like scikit-learn,

```python
import numpy as np
from sklearn.datasets import load_boston

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import StandardScaler

from mla_sani.supervised.linear_model import LinearRegression
from mla_sani.model_selection import train_test_split
from mla_sani.preprocessing import StandardScaler
from mla_sani.metrics import mean_absolute_error

data = load_boston()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print(mean_absolute_error(y_test, y_pred))
```

while the DL algorithms are implemented in the way that it just looks like simplified Keras

```python
import numpy as np
from sklearn.datasets import load_digits

from mla_sani.model_selection import train_test_split
from mla_sani.metrics import confusion_matrix

from mla_sani.nn.layers import Input, Conv2D, Activation, Dropout, Flatten, Dense
from mla_sani.nn.models import Sequential
from mla_sani.nn.optimizers import Adam
from mla_sani.losses import CategoricalCrossEntropy

data = load_digits()
X, y = data.data.reshape(-1, 8, 8, 1), data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

cnn = Sequential()
cnn.add(Input(X.shape[1:]))
cnn.add(Conv2D(16, (3, 3), padding='same'))
cnn.add(Activation('relu'))
cnn.add(Dropout(rate=0.1))
cnn.add(Flatten())
cnn.add(Dense(10))
cnn.add(Activation('softmax'))

cnn.compile(optimizer=Adam(), loss=CategoricalCrossEntropy(labels=np.unique(y)))
cnn.fit(X_train, y_train, epochs=30, batch_size=128)
y_pred = cnn.predict(X_test).argmax(axis=1)
print(confusion_matrix(y_test, y_pred))
```

Hopefully, this project could help some engineers who are not that good at math,  but know coding well and are tring to grasp these algrithms as quick as possible.


