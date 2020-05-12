import numpy as np
import matplotlib.pyplot as plt
X = [
        [10],
        [20],
        [30],
    ]
y = [
        [11],
        [21],
        [32],
    ]


X = np.array(X)
y = np.array(y)

# print(np.matmul(X, y))
# print(np.dot(X.T, X))
# print(np.matrix(np.dot(X.T, X)).I)
# print(np.dot(X.T, y))
# print(np.matrix(np.dot(X.T, X)).I * np.dot(X.T, y))
# print(np.dot(np.matrix(np.dot(X.T, X)).I, np.dot(X.T, y)))
print(np.matmul(np.matrix(np.dot(X.T, X)).I, np.dot(X.T, y)))

v = np.matmul(np.matrix(np.dot(X.T, X)).I, np.dot(X.T, y))


# plt.scatter(x=X,y=y,c='g')
# plt.show()