from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


clf = clf.fit(X, Y)

prediction = clf.predict([[190, 90, 47]])

print(prediction)

# 1 QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis()
clf = clf.fit(X, Y)

prediction = clf.predict([[159, 55, 37]])
print(prediction)


# 2 KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10)
neigh = neigh.fit(X, Y)

prediction = neigh.predict([[160, 60, 38]])
print(prediction)

# 3 RandomForestClassifier

clf = RandomForestClassifier(n_estimators=55)
clf = clf.fit(X, Y)

prediction = clf.predict([[175, 64, 39]])
print(prediction)



# Accuracy

y_pred = ['female', 'female', 'female', 'female']
y_true =  ['female', 'female', 'female', 'female']

print(accuracy_score(y_true, y_pred))