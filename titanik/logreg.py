import numpy as np
import pandas as pd

def add_intercept(x):
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def ucitaj_csv(ime_fajla):

    # koristimo pandas za importovanje csv fajla
    procitan_csv = pd.read_csv(ime_fajla)

    # izvlacimo pandas dataframe-ove sa odgovarajucim vrednostima
    x_train = pd.DataFrame(procitan_csv, columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Sex'])
    
    # ispunjavamo polja koja nedostaju srednjim vrednostima
    x_train['Pclass'].fillna(value = int(x_train['Pclass'].mean()), inplace=True)
    x_train['Age'].fillna(value = int(x_train['Age'].mean()), inplace=True)
    x_train['SibSp'].fillna(value = int(x_train['SibSp'].mean()), inplace=True)
    x_train['Parch'].fillna(value = int(x_train['Parch'].mean()), inplace=True)
    
    # postavljamo za polje pola numericke vrednosti
    x_train = x_train.replace(to_replace='female', value=0)
    x_train = x_train.replace(to_replace='male', value=1)
    x_train['Sex'].fillna(value = int(x_train['Sex'].mean()), inplace=True)

    y_train = pd.DataFrame(procitan_csv, columns = ['Survived']).to_numpy()

    # vracamo nizove pretvorene u numpy array
    return add_intercept(x_train.to_numpy()), y_train


def main(train_path, valid_path, save_path):

    x_train, y_train = ucitaj_csv(train_path)


    print("Logisticka regresija njutnovom metodom nad: ", train_path)

    clf = LogisticRegression(theta_0=np.zeros((len(x_train[0]),1)))
    clf.fit(x_train, y_train)

    x_valid, y_valid = ucitaj_csv(valid_path)

    export = pd.DataFrame(pd.read_csv(valid_path), columns = ['PassengerId'])

    export.to_csv(save_path, index=False)

    predvidjanja = clf.predict(x_valid)

    export["Survived"] = predvidjanja

    export.to_csv(save_path, index=False)

class LogisticRegression:
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):

        print(self.theta)

        def g(z):
            return np.true_divide(1,(1+np.exp(-z)))

        def h(x_i):
            return g(np.dot(self.theta.transpose(), x_i))

        def l(theta):
            sum = 0
            for i in range(y.size):
                htheta = h(x[i])
                if htheta == 1:
                    htheta -= 1e-16 # logaritam nule
                sum += y[i] * np.log(htheta) + (1 - y[i]) * np.log(1 - htheta)
            return sum
        
        def l_izvod(j):
            sum = 0
            for i in range(y.size):
                sum += (y[i] - h(x[i])) * x[i][j]
            return sum
	
        def l_2izvod(i, j):
            sum = 0
            for k in range(y.size):
                htheta = h(x[k])
                sum += (x[k][i] * x[k][j] * htheta) * (1 - htheta)
            return sum

        def gradijent():
            grad = np.zeros((self.theta.size,1))
            for i in range(grad.size):
                grad[i] = l_izvod(i)
            return grad

        def hesijan():
            hesijan = np.zeros((self.theta.size, self.theta.size))
            for i in range(self.theta.size):
                for j in range(self.theta.size):
                    hesijan[i][j] = -l_2izvod(i, j)
            return hesijan
	
        def trening():
            grad = gradijent() 
            h = hesijan()
            self.theta = self.theta - np.dot(np.linalg.inv(h), grad)

        i = 0
        dtheta = 1
        while i < self.max_iter and abs(dtheta) > self.eps:
            thetastaro = self.theta
            trening()
            thetanovo = self.theta
            dtheta = np.linalg.norm(thetastaro - thetanovo)
            print("Novo theta: ", self.theta)
            i += 1
            print("Iteracija: %d" % i)

        print("Trening zavrsen u: %d iteracija, promena thete je %.10f; " % (i,abs(dtheta)))
        print("Konacno:", self.theta)


    def predict(self, x):
        pogadjanja = []

        def g(z):
            return 1/(1+np.exp(-z))

        def h(x_i):
            return g(np.dot(self.theta.transpose(),x_i))
        
        for i in x:
            if h(i) < 0.5:
                pogadjanja.append(0)
            else:
                pogadjanja.append(1)
        
        return pogadjanja        

if __name__ == '__main__':
    main(train_path='train.csv',
         valid_path='test.csv',
         save_path='submission.csv')
