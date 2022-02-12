import numpy as np
import util


def main(train_path, valid_path, save_path, plot_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier

    print("Logisticka regresija nad: ", train_path, "\n")

    clf = LogisticRegression(theta_0=np.zeros((3,1)))
    clf.fit(x_train, y_train)
    
    # Plot decision boundary on top of validation set set
    
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, clf.theta, plot_path)
    
    # Use np.savetxt to save predictions on eval set to save_path
    
    pogadjanja = clf.predict(x_valid)
    np.savetxt(save_path, pogadjanja)
    
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

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

	    # konstrukcija matrica gradijenta i hesijana

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
	
        # definisanje iteracije treninga	

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
	
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        pogadjanja = []


        def g(z):
            return 1/(1+np.exp(-z))

        def h(x_i):
            return g(np.dot(self.theta.transpose(),x_i))
        
        for i in x:
            pogadjanja.append(h(i))
        
        return pogadjanja
        
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt',
         plot_path='logreg_plot_1.png')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt',
         plot_path='logreg_plot_2.png')
