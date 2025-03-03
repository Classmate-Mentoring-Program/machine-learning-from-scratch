import numpy as np
import matplotlib.pyplot as plt

# Regression Lineaire - methode directe en utilisant l'equation normale
class RL_closedForm:

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        line, column = X.shape
        self.bias = np.ones((line, 1))

        X_b = np.c_[self.bias, X] # ajouter le biais aux instances de X
        transposee = X_b.T
        # Equation normale
        self.weights = np.linalg.inv(transposee.dot(X_b)).dot(transposee).dot(y)

    def predict(self, X):
        bias = np.ones((X.shape[0], 1))
        X_b = np.c_[bias, X]
        y_pred = X_b.dot(self.weights)
        return y_pred
    
    def rmse(self, y, y_pred):
        error = np.mean((y - y_pred)**2) * 100
        return error

# Regression Lineaire - methode iterative avec la descente de Gradient
class RL:

    def __init__(self, lrnRate= 0.1, nIter=100):
        self.lrnRate = lrnRate
        self.nIter = nIter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        line, column = X.shape

        # initialisation des parametres
        self.weights = np.zeros(column).reshape(1, -1)
        self.bias = 0

        # algorithme de descente de gradient
        for i in range(self.nIter):
            
            y_h = np.dot(X, self.weights) + self.bias

            dw = ((2/line)) * np.dot(X.T, (y_h - y))
            db = ((2/line)) * np.sum((y_h - y))

            self.weights -= self.lrnRate * dw
            self.bias -= self.lrnRate * db
            print(f">> Iteration {i}, RMSE: {self.rmse(y, y_h)}")
    
    def rmse(self, y, y_pred):
        error = np.mean((y - y_pred)**2) * 100
        return error

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

# regression polynomiale avec algorithme de la descnte de gradient
class PL:

    def __init__(self, degree= 2, lrnRate= 0.01, nIter=100):
        self.degree = degree
        self.lrnRate = lrnRate
        self.nIter = nIter
        self.weights = None
        self.bias = None
    
    def _transform(self, X):
        X_poly = X.copy()

        for i in range(2, self.degree +1):
            X_poly = np.hstack((X_poly, X**i))
        return X_poly
    
    def fit(self, X, y):
        X = self._transform(X)
        line, column = X.shape

        # initialisation des parametres
        self.weights = np.zeros((column,1))
        self.bias = 0

        # algorithme de descente de gradient
        for i in range(self.nIter):
            
            y_h = np.dot(X, self.weights) + self.bias

            dw = ((2/line)) * np.dot(X.T, (y_h - y))
            db = ((2/line)) * np.sum((y_h - y))

            self.weights -= self.lrnRate * dw
            self.bias -= self.lrnRate * db
            print(f">> Iteration {i}, RMSE: {self.rmse(y, y_h)}")
    
    def rmse(self, y, y_pred):
        error = np.mean((y - y_pred)**2) * 100
        return error

    def predict(self, X):
        X = self._transform(X)
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    

if __name__ == "__main__":
    """
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    reg1 = RL_closedForm()
    reg1.fit(X, y)
    y_pred = reg1.predict(X)
    plt.plot(X, y, "r.")
    plt.plot(X, y_pred, "b-")
    plt.title("Equation normale")
    plt.show()
    print(">> Erreur 1: ", reg1.rmse(y, y_pred))
    reg2 = RL()
    reg2.fit(X, y)
    y_pred = reg2.predict(X)
    plt.plot(X, y, "r.")
    plt.plot(X, y_pred, "b-")
    plt.title("Descente de Gradient")
    plt.show()
    print(">> Erreur 2: ", reg2.rmse(y, y_pred))"
    """
    X = 2 * np.random.rand(100, 1)
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
    poly = PL(degree=3)
    poly.fit(X, y)
    y_pred = poly.predict(X)
    plt.plot(X, y, "r.")
    plt.plot(X, y_pred, "b-")
    plt.title("Polynomial Regression")
    plt.show()
    print(">> Erreur 2: ", poly.rmse(y, y_pred))