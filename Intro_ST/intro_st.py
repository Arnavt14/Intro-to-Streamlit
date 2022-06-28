# import streamlit as st
# import pandas as pd
# from sklearn import datasets
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
#
# # using st.title we can create the title of the app
# st.title("Streamlit Example")
#
# #using st.write to just write some text.
# #using markup language we can create headings as well using st.write.
#
# st.write("""
# # Explore different Classifier
# Which one is the best?
# """)
#
# # st.write("Welcome")
# # st.write("""
# # # Heading one
# # Text below the heading
# # """)
#
# # # using st.button we can add a button. we type text in the bracket to display it on the button. we can also use on_click to decide what happends when we click the button
# #
# # st.button("Login")
# # st.button("SignUp")
# #
# # #using st.file_uploader we can upload a file
# # file = st.file_uploader("Pick a file")
#
# # we can use widgets as well
# # for eg. we can use a selectbox for selecting items maybe datasets etc. The first the text is displayed over the selectboc. The tuple contains the options.
# # we can also assign the box to a variable.
# # we can also put the sslectbox in the sidebar as well.
# #then we can show the name of the dataset selected
# data_set = st.sidebar.selectbox("Datasets", {"Iris","Wine","Breast Cancer"})
# classifier = st.sidebar.selectbox("Classifier", {"KNN","SVM","Random Forest"})
#
# def get_dataset (data_set):
#     if data_set == "Iris":
#         data = datasets.load_iris()
#     elif data_set == "Breast Cancer":
#         data = datasets.load_breast_cancer()
#     else:
#         data = datasets.load_wine()
#     x = data.data
#     y = data.target
#     return x,y
#
# x,y = get_dataset(data_set)
# st.write("Shape of dataset", x.shape)
# st.write("Number of classes",len(np.unique(y)))
#
# def add_parameter_ui (clf_name):
#     params = dict()
#     if clf_name == "KNN":
#         # using slider, we can create a slider type widget which is useful in selecting a value
#         K = st.sidebar.slider("K", 1, 15)
#         params["K"] = K
#     elif clf_name == "SVM":
#         C = st.sidebar.slider("C", 0.01, 10.0)
#         params["C"] = C
#     else:
#         max_depth = st.sidebar.slider("Max Depth", 2,15)
#         n_estimators = st.sidebar.slider("n_estimators", 1, 100)
#         params["max_depth"] = max_depth
#         params["n_estimators"] = n_estimators
#     return params
#
# params = add_parameter_ui(classifier)
#
# # Now all that's left is to setup the classifier
#
# def get_clf (clf_name, params):
#     if clf_name == "KNN":
#         clf = KNeighborsClassifier(n_neighbors = params["K"])
#     elif clf_name == "SVM":
#         clf = SVC(C = params["C"])
#     else:
#         clf = RandomForestClassifier(n_estimators = params["n_estimators"], max_depth = params["max_depth"], random_state = 1234)
#
#     return params
#
# clf = get_clf(classifier, params)
#
# # performing classification
#
# X_train, X_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state = 1234)
#
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# st.write(f"classifier = {classifier}")
# st.write(f"accuracy = {accuracy}")
#
# # plot


import streamlit as st
import numpy as np


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title('Streamlit Example')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)
