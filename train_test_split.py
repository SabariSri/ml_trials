import numpy as np
from sklearn import model_selection, datasets

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)

print(diabetes_X)
print(diabetes_y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(diabetes_X['age'], diabetes_y, train_size=.9)
print(X_train)
print("X TEST", X_test)
