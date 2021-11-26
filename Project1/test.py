import numpy as np

one = False

if one == True:
    x = np.random.randn(5,4)
    N, D = x.shape
    w1 = np.random.randn(4,10)
    b1 = np.zeros((10,))
    w2 = np.random.randn(10,3)
    b2 = np.zeros((3,))
    z1 = np.array([np.dot(x[i], w1) + b1 for i in range(N)])
    p1 = np.maximum(0, z1)
    z2 = np.array([np.dot(p1[i], w2) + b2 for i in range(N)])
    N, C = z2.shape
    y = [0,1,2,2,1]
    reg = 0.01
    p2 = np.array([[np.exp(z2[i][j]) / np.sum(np.exp(z2[i])) for j in range(C)] for i in range(N)])
    dl = 0
    for i in range(N):
        dl += -np.log(p2[i][y[i]])
    dl /= N
    R = (np.sum(np.square(w1)) + np.sum(np.square(w2))) * reg
    loss = dl + R
    new_y = np.zeros((N, C))
    for i in range(N):
        new_y[i][y[i]] = 1
    dz2 = p2 - new_y
    dw2 = np.dot(p1.T, dz2)
    grads = {}
    grads['W2'] = dw2
    grads['b2'] = dz2
    dp1 = np.dot(dz2, w2.T)
    dz1 = dp1.copy()
    a, b = dz1.shape
    for i in range(a):
        for j in range(b):
            if dz1[i][j] <= 0:
                dz1[i][j] = 0
    dw1 = np.dot(x.T, dz1)
    grads['W1'] = dw1
    grads['b1'] = dz1
    print(grads)

x = np.random.randn(5,4)
N, D = x.shape
idx = np.random.choice(5, 10)
x_batch = x[np.random.choice(5,10)]
x_batch2 = np.array([np.random.choice(x[i], 100) for i in range(N)])
scores = np.array([np.random.randn(3) for i in range(N)])
y_pred = np.array([np.argmax(scores[i]) for i in range(N)])
print(scores)
print(y_pred)