import numpy as np
import util

def parametri_normalizacije(x, n, indexx1):
    sumx1 = 0
    sumx2 = 0

    najvecix1 = x[0][indexx1]
    najmanjix1 = x[0][indexx1]
    najvecix2 = x[0][indexx1+1]
    najmanjix2 = x[0][indexx1+1]

    for i in x:
        sumx1 += i[indexx1]
        sumx2 += i[indexx1+1]
        if i[indexx1] > najvecix1:
            najvecix1 = i[indexx1]
        if i[indexx1] < najmanjix1:
            najmanjix1 = i[indexx1]
        if i[indexx1+1] > najvecix2:
            najvecix2 = i[indexx1+1]
        if i[indexx1+1] < najmanjix2:
            najmanjix2 = i[indexx1+1]

    prosecnox1 = sumx1/n
    prosecnox2 = sumx2/n

    print("Formula normalizacije: (x1 - ", prosecnox1, ") / ", najvecix1-najmanjix1)
    print("Formula normalizacije: (x2 - ", prosecnox2, ") / ", najvecix2-najmanjix2)

    return (prosecnox1, najmanjix1, najvecix1, prosecnox2, najmanjix2, najvecix2)


def normalizuj(x, n, prosecnox1, najmanjix1, najvecix1, prosecnox2, najmanjix2, najvecix2, indexx1):
    for i in range(n):
        x[i][indexx1] = (x[i][indexx1] - prosecnox1)/(najvecix1-najmanjix1)
        #print("prethodno x2: ",x[i][indexx1+1])
        x[i][indexx1+1] = (x[i][indexx1+1] - prosecnox2)/(najvecix2-najmanjix2)
        #print("novo x2: ", x[i][indexx1+1])
    return x




def main(train_path, valid_path, save_path, plot_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    
    # Train a GDA classifier

    clf = GDA(theta_0=np.zeros((3,1)))

    prosecnox1, najmanjix1, najvecix1, prosecnox2, najmanjix2, najvecix2 = parametri_normalizacije(x_train, len(y_train), 0)

    x = normalizuj(x_train, len(y_train), prosecnox1, najmanjix1, najvecix1, prosecnox2, najmanjix2, najvecix2, 0)

    clf.fit(x_train, y_train)
    
    # Plot decision boundary on validation set
    
    x_valid, y_valid = util.load_dataset(valid_path)
    x_valid = normalizuj(x_valid, len(y_valid), prosecnox1, najmanjix1, najvecix1, prosecnox2, najmanjix2, najvecix2, 0)
    util.plot(x_valid, y_valid, clf.theta, plot_path)
    
    # Use np.savetxt to save outputs from validation set to save_path
    
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_valid = normalizuj(x_valid, len(y_valid), prosecnox1, najmanjix1, najvecix1, prosecnox2, najmanjix2, najvecix2, 1)
    pogadjanja = clf.predict(x_valid)
    np.savetxt(save_path, pogadjanja)
    
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        phi = 0
        mu_0_gore = np.zeros((1,2))        
        mu_0_dole = 0        
        mu_1_gore = np.zeros((1,2))
        mu_1_dole = 0
 
        for i in range(len(y)):
            if y[i] == 0:
                mu_0_gore += x[i]
                mu_0_dole += 1
            else:
                phi += 1
                mu_1_gore += x[i]
                mu_1_dole += 1

        phi = phi / len(y)
        mu_0 = mu_0_gore / mu_0_dole
        mu_1 = mu_1_gore / mu_1_dole
        
        sigma = np.zeros((2,2))
        
        for i in range(len(y)):
            if y[i] == 0:
                sigma += np.dot((x[i] - mu_0).transpose(), x[i] - mu_0)
            else:
                sigma += np.dot((x[i] - mu_1).transpose(), x[i] - mu_1)

        sigma = sigma / len(y)
        sigma_inv = np.linalg.inv(sigma)

        print("phi:", phi, "\nmu_0:", mu_0, "\nmu_1:", mu_1, "\nsigma:", sigma)
        
        # Write theta in terms of the parameters

        self.theta[0] = - (-(1/2)*(np.dot(np.dot(mu_0,sigma_inv),mu_0.transpose()) - np.dot(np.dot(mu_1,sigma_inv),mu_1.transpose())) - np.log((1 - phi)/phi)) 
        thetadvadela = - np.dot(mu_0 - mu_1,sigma_inv)
        self.theta[1] = thetadvadela[0][0]
        self.theta[2] = thetadvadela[0][1]

        print("Theta: ", self.theta) 
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        pogadjanja = []
        
        for i in x:
            pogadjanja.append(1/(1 + np.exp(-np.dot(self.theta.transpose(), i))))
        return np.array(pogadjanja)

        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_mod_1.txt',
         plot_path='gda_plot_mod_1.png')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_mod_2.txt',
         plot_path='gda_plot_mod_2.png')
