from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
def begin():

    ##train test set creation
    train_feat = np.load('train_set_x.npy')
    train_label = np.load('train_set_y.npy')
    train_label=train_label.reshape(500,1)
    X_train = train_feat[:350]
    X_test = train_feat[350:]

    y_train = train_label[:350]
    y_train = y_train.T

    y_test = train_label[350:]
    y_test = y_test.T

    m_train = X_train.shape[0]
    m_test = X_test.shape[0]
    num_px = train_feat.shape[1]

    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(X_train.shape))
    print ("train_set_y shape: " + str(y_train.shape))
    print ("test_set_x shape: " + str(X_test.shape))
    print ("test_set_y shape: " + str(y_test.shape))

    # Reshape the training and test examples
    X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
    X_test_flatten = X_test.reshape(X_test.shape[0], -1).T

    print ("X_train_flatten shape: " + str(X_train_flatten.shape))
    print ("y_train: " + str(y_train.shape))
    print ("X_test_flatten shape: " + str(X_test_flatten.shape))
    print ("y_test shape: " + str(y_test.shape))
    print ("sanity check after reshaping: " + str(X_train_flatten[0:5,0]))

    #Let's standardize our dataset.
    #image data can be standardize by dividing data by 255
    #images are usually encoded with 8 bit encoding
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.

    ## Activation function
    def sigmoid(z):
        """
        Compute the sigmoid of z

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """
        
        s = 1 / (1 + np.exp(-z))   
        return s

    def initialize_parameters(dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        
        w = np.zeros(shape=(dim, 1))
        b = 0

        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        
        return w, b

    ## forward and backward propagation
    def propagate(w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        
        Tips:
        - Write your code step by step for the propagation
        """
        
        m = X.shape[1]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        A = sigmoid(np.dot(w.T, X) + b)  # compute activation
        cost = (- 1 / m) * (np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))) # compute cost
        
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                 "db": db}
        
        return grads, cost

    ## optimization of the variables
    def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        
        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """
        
        costs = []
        
        for i in range(num_iterations):
            
            
            # Cost and gradient calculation
           
            grads, cost = propagate(w, b, X, Y)
            
            
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            # update rule
            w = w - learning_rate * dw  # need to broadcast
            b = b - learning_rate * db
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))
        
        params = {"w": w,
                  "b": b}
        
        grads = {"dw": dw,
                 "db": db}
        
        return params, grads, costs

    ## predictions
    def predict(w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = sigmoid(np.dot(w.T, X) + b)
       
        
        for i in range(A.shape[1]):
            # Convert probabilities a[0,i] to actual predictions p[0,i]
            
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
            
        
        assert(Y_prediction.shape == (1, m))
        
        return Y_prediction

    ## implement model
    def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        """
        Builds the logistic regression model by calling the function you've implemented previously
        
        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations
        
        Returns:
        d -- dictionary containing information about the model.
        """
        
        # initialize parameters with zeros.
        w, b = initialize_parameters(X_train.shape[0])

        # Gradient descent
        parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        
        # Retrieve parameters w and b from dictionary "parameters"
        w = parameters["w"]
        b = parameters["b"]
        
        # Predict test/train set examples.
        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)


        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        
        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}
        
        return d
    d = model(X_train, y_train, X_test, y_test, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

    def real_predictions():
        real_img_path = raw_input('give your image to clasify along with its path: ')
        img_real = cv2.imread(real_img_path)
        img_real = cv2.resize(img_real,(64,64))
        plt.imshow(img_real)
        img_real=img_real.reshape((1,64*64*3)).T
                
        res = predict(d['w'],d['b'],img_real)
        
        if res == 0:
            print 'I think it is not a cat'
            plt.show()
        else:
            print 'It is a cat'
            plt.show()

    while(True):
        choice = raw_input('want to give your image to classify?y/n: ')
        if choice == 'y':
            real_predictions()
        else:
            break
############ END OF PROGRAM ##############

if __name__ == '__main__':
    begin()
