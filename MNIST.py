import numpy as np  # noqa: I001
np.random.seed(0)
#aici incarc datasetul MNIST
from sklearn.datasets import fetch_openml  # noqa: I001
X,y = fetch_openml("mnist_784" , version=1 , return_X_y=True, as_frame=False, parser="liac-arff")

#Astea sunt principalele functii pe care le folosesc(o sa mai adaug cateva don worry)
def ReLU(x):
    return np.maximum (x,0)
def Softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
def Onehot(x):
    raspuns = np.zeros(10)
    raspuns[x] = 1
    return raspuns
def Cross_entropy(p,t):
    L = -(t@np.log(p + 1e-12))
    return L

#functia asta de aici dezactiveaza e ul acela care apare la calcule foarte mici sau foarte mari 
np.set_printoptions(suppress= True)

#Prelucrarea datelor 
X = X/255
y = y.astype(int)
x_antrenare = X[:59999]
x_test = X[60000:]
y_antrenare = y[:59999]
y_test = y[60000:]

W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

#praaaaactic aici este faza de antrenare practic

lr = 0.01

for epoca in range (10):

    total = 0

    for i in range (59999):

        x_antrenament = x_antrenare[i]
        y_antrenament = y_antrenare[i]

        z1 = (W1 @ x_antrenament) + b1
        a1 = ReLU(z1)

        z2 = (W2 @ a1) + b2

        p = Softmax(z2)
        t = Onehot(y_antrenament)
        cost = Cross_entropy(p,t)

        total += cost

        #Aici vine backward passul !!! lr e learning rate nu stiu cum sa scriu niu in vscode :(


        ds2 = p - t
        dW2 = np.outer(ds2,a1)
        db2 = ds2

        ds1 = (W2.T@ds2)*(z1>0)
        dW1 = np.outer(ds1,x_antrenament)
        db1 = ds1

        W1 -= lr*dW1
        b1 -= lr*db1
        W2 -= lr*dW2
        b2 -= lr*db2

    print (total)

np.save("W1.npy",W1)
np.save("b1.npy",b1)
np.save("W2.npy",W2)
np.save("b2.npy",b2)
    

