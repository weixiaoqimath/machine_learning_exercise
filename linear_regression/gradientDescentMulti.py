import numpy as np
import computeCostMulti 

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
        Perform gradient descent for multivariable linear regression.
    """
    m = X.shape[0] # number of training examples
    J_history = np.zeros(num_iters);
    
    i = 0
    print("Now we begin training. The learning rate is {} and number of iteration is {}.".format(alpha, num_iters))
    while i < num_iters:
        # Perform a single gradient step on the parameter vector theta. 
        theta = theta - alpha/m * np.transpose(X)@(X@theta-y);
        # Save the cost J in every iteration    
        J_history[i] = computeCostMulti.computeCostMulti(X, y, theta)
        if (i+1) % 10 == 0:
            print("After {} iterations, the loss is {:.4f}.".format(i+1, J_history[i]))   
        i += 1
        
    return [theta, J_history]






