#!/usr/bin/env python
# coding: utf-8

# ## Universal Representation Theorem - Gradient Descent Optimisation
# 
# Here we study the possibility to represent functions with a MLP with a single hidden layer (n1 hidden units).
# As activation functions, we use the sigmoid ('logit') function.
# 
# Then, we generate training data - by assuming a function on the unit interval [0,1]. Here, we provide to families of functions:
# * Beta distribution function: $b_{\alpha,\beta}(x)=x^\alpha\cdot(1-x)^\beta$
# * Sine function: $sin_\omega(x)=\sin(2\pi\omega\cdot x)$
# 
# Finally, we use mini-batch-gradient descent to minimize MSE cost.
# 
# Goals:
# * Learn how a given function can be represented with a single layer MLP;
# * Understand that, in principle, it can be learned from sample data;
# * Understand that the optimization by using plain gradient (MBGD) is not always straightforward; 
# * Experience that the choice of the hyper-parameters number of hidden units, batchsize, learning rate is tricky. 
# 

# #### Plot Utility

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_function(x,y):
    plt.plot(x, y)
    plt.xlabel('x')
    plt.show()
    
def plot_compare_function(x,y1,y2, label1='', label2=''):
    plt.plot(x, y1, label=label1)
    plt.xlabel('x')
    plt.plot(x, y2, label=label2)
    if label1 and label2:
        plt.legend()
    plt.show()


# ### Model

# In[ ]:


def sigmoid(z):
      return 1. / (1. + np.exp(-z))


# In[ ]:


def predict(X,W1,b1,W2,b2):
    """
    Computes the output for the single hidden layer network (n1 units) with 1d input and 1d output.
    
    Arguments:
    W1 -- weights of the hidden layer with shape (n1,1)
    b1 -- biases of the hidden units with shape (n1,1)
    W2 -- weights of the output layer with shape (1,n1)
    b2 -- bias of the output
    X  -- input data with m samples and shape (1,m)
    
    Returns:
    A2 -- Output from the network of shape (1,m) 
    
    """
    
    
    A2 = sum(np.dot(W2, sigmoid(np.dot(W1,X)+b1)) + b2).reshape(1,X.shape[1])
    
    
    return A2


# #### TEST - Prediction

# In[ ]:


W1 = np.array([0.4,0.2,-0.4]).reshape(3,1) # n1 = 3
b1 = np.array([0.1,0.1,0.1]).reshape(3,1)
W2 = np.array([1,2,1]).reshape(1,3)
b2 = -1
X = np.linspace(-1,1,5).reshape((1,5))
Ypred = predict(X,W1,b1,W2,b2)
Yexp = np.array([0.99805844, 1.04946333, 1.09991675, 1.14913132, 1.19690185]).reshape(1,5)
np.testing.assert_array_almost_equal(Ypred,Yexp,decimal=8)


# #### Cost

# In[ ]:


def cost(X,Y,W1,b1,W2,b2):
    
    
    m = np.size(Y, 1)
    
    cost = 1/(2*m) * np.sum(np.square(Y - (sum(np.dot(W2, sigmoid(np.dot(W1,X) + b1)) + b2))))

    return cost


# #### TEST - Cost

# In[ ]:


W1 = np.array([4,5,6]).reshape(3,1)
W2 = np.array([1,2,3]).reshape(1,3)
b1 = np.array([1,1,1]).reshape(3,1)
b2 = 2
X = np.linspace(-1,1,5).reshape(1,5)
Y = 2.0*np.ones(5).reshape(1,5)
c = cost(X,Y,W1,b1,W2,b2)
cexp = 9.01669099
np.testing.assert_almost_equal(c,cexp,decimal=8)


# #### Gradient

# In[ ]:


def gradient(W1,b1,W2,b2,X,Y):
   
    m = Y.size
    
    z=np.dot(W1,X)+b1
   
    
    #W1
    first_part=(Y - (sum(np.dot(W2, sigmoid(z)) + b2)))
    l1=sigmoid(z)
    l2=1-sigmoid(z)
    second_part=l1*l2
    together=first_part*second_part
    dW1=(-1/m) * sum(np.transpose(np.dot(together,np.transpose(X)))*W2)
    
    dW1=dW1.reshape(W1.shape[0],1)
    
    #b1
    
    first_part=(Y - (sum(np.dot(W2, sigmoid(z)) + b2)))
    l1=sigmoid(z)
    l2=1-sigmoid(z)
    second_part=l1*l2
    together=first_part*second_part
    db1=(-1/m) * sum(together.sum(1) * W2)
    db1=db1.reshape(W1.shape[0],1)
    
    #W2
    first_part=(Y - (sum(np.dot(W2, sigmoid(z)) + b2)))
    second_part=sigmoid(z)
    together=first_part*second_part
    dW2=(-1/m) * together.sum(1)
    dW2=dW2.reshape(1,W2.shape[1])
    
    #b2
    db2=(Y - (sum(np.dot(W2, sigmoid(z)) + b2)))
    db2=(-1/m) * db2.sum()

    
    
    return {'dW1':dW1, 'dW2':dW2, 'db1':db1, 'db2':db2}


# #### TEST - Gradient

# In[8]:


W1 = np.array([4,5,6]).reshape(3,1)
W2 = np.array([1,2,3]).reshape(1,3)
b1 = np.array([1,1,1]).reshape(3,1)
b2 = 2
X = np.array([1,2,3,4,5,6,7]).reshape((1,7))
Y = np.array([2,2,2,2,2,2,2]).reshape((1,7))
gradJ = gradient(W1,b1,W2,b2,X,Y)
dW1exp = np.array([0.00590214,0.00427602,0.00234663]).reshape(W1.shape)
db1exp = np.array([0.00579241,0.004247,0.00234079]).reshape(b1.shape)
dW2exp = np.array([5.99209251,5.99579451,5.99714226]).reshape(W2.shape)
db2exp = 5.99792323
np.testing.assert_array_almost_equal(gradJ['dW1'],dW1exp,decimal=8)
np.testing.assert_array_almost_equal(gradJ['db1'],db1exp,decimal=8)
np.testing.assert_array_almost_equal(gradJ['dW2'],dW2exp,decimal=8)
np.testing.assert_almost_equal(gradJ['db2'],db2exp,decimal=8)

print(dW1exp)
print(gradJ['dW1'])


# #### Training Loop

# In[ ]:


def train(X,Y,n1,nepochs,batchsize=32,learning_rate=0.1,seed = 10):
   
    # initialize weights
    #np.random.seed(seed)
    W1 = np.random.uniform(-1,1,n1).reshape(n1,1)*0.05
    b1 = np.zeros((n1,1),dtype='float')
    W2 = np.random.uniform(-1,1,n1).reshape(1,n1)*0.05
    b2 = 0.0
    
    m = X.shape[1]
    mb = int(m/batchsize)
    indices = np.arange(m)
    #np.random.shuffle(indices)
    
    # remember the epoch id and cost after each epoch for constructing the learning curve at the end
    costs = [] 
    epochs = []
    


    # Initial cost value:
    epochs.append(0)
    costs.append(cost(X,Y,W1,b1,W2,b2)) 
    
    # training loop
    for epoch in range(nepochs):

      
        
        #TODO: Loop over Batches
        for j in range(0, m, batchsize):
            #Batch selection
            #rows_train=np.random.choice(a=m,size=mb)
            X_train = X[:,j:j+batchsize]
            Y_train = Y[:,j:j+batchsize]

            # Cost function specific part
            c = cost(X_train,Y_train,W1,b1,W2,b2)

            gradJ = gradient(W1=W1,b1=b1,W2=W2,b2=b2,X=X_train,Y=Y_train)


            # Updating parameter w & b
            W1 = W1 - learning_rate * gradJ['dW1']
            W2 = W2 - learning_rate * gradJ['dW2']
            b1 = b1 - learning_rate * gradJ['db1']
            b2 = b2 - learning_rate * gradJ['db2']
        
      
        epochs.append(epoch+1)
        costs.append(cost(X_train,Y_train,W1,b1,W2,b2))        
    
    print(costs[-1])    
    params = {'W1':W1, 'W2':W2,'b1':b1,'b2':b2}    
    return params, np.array(epochs), np.array(costs)


# #### TEST - No simple test for the training loop....

# ### Generation of the Training Data 

# In[ ]:


def beta_fct(x,alpha,beta):
    """
    Parameters:
    x - input array
    alpha, beta -- larger values lead to more pronounced peaks
    """
    c = alpha/(alpha+beta)
    norm = c**alpha*(1-c)**beta
    return x**alpha*(1-x)**beta/norm


# In[ ]:


def sin_fct(x,omega):
    """
    Parameters:
    x -- input array
    omega -- frequency (~number of cycles within the unit interval)
    """
    return np.sin(x*2*np.pi*omega)


# In[ ]:


def generate_inputs(m, func, random=True, vargs=None):
    """
    Generates m (x,y=f(x))-samples by either generating random x-values in the unit interval (random=True) or by 
    generating a grid of such values. Then the y values (used as labels below) are created from the function object 
    `func`.
    Parameter needed to define the function `func` can be passed as vargs-dict. 
    """
    if random:
        x = np.random.rand(1,m)
        y = func(x, **vargs)
    else:
        x = np.linspace(0,1,m).reshape(1,m)
        y = func(x,**vargs)
    return x,y


# In[ ]:


m = 1000
func = beta_fct
vargs={'alpha':2.0,'beta':2.0}
#func = sin_fct
#vargs={'omega':3.0}

X,Y = generate_inputs(m,func,vargs=vargs)


# In[ ]:


plt.plot(X[0,],Y[0,],'+')
plt.show()


# ### Normalize the Input and Output
# 
# It turns out that it is important to normalize the input and the output data here.
# Remember the mu's and sigma's so that you can also apply it to the test data below!

# In[ ]:


def normalize(X, mu=None, stdev=None):
    """
    Normalizes the data X. If not provided, mu and sigma is computed.
    
    Returns:
    X1 -- normalized data (array of the same shape as input)
    mu -- mean
    stdev -- standard deviation
    """
  
    X1=(X-np.mean(X))/np.std(X)
    
    mu=np.mean(X)
    
    stdev=np.std(X)
   
    
    return X1,mu,stdev


# In[ ]:


def inv_normalize(X1, mu, stdev):
    """
    Invert the normalization.

    Returns:
    X -- unnormalized data (array of the same shape as input X1)
    """
    
    
    X=(X1*stdev+mu)
   
    
    return X


# In[ ]:


# Input Normalization (average muX and standard deviation stdevX)
X1, muX, stdevX = normalize(X)

# Output Normalization (average muY and standard deviation stdevY)
Y1, muY, stdevY = normalize(Y)


# In[ ]:


plt.plot(X1[0,:],Y1[0,:],'+')


# ### Perform the Training

# In[ ]:


# Use the normalized inputs and outputs
n1 = 32 # number of hidden units
nepochs = 1000 # number of epochs
batchsize = 10
learning_rate = 0.1

### START YOUR CODE ###
params, epochs, costs=train(X=X1,
                            Y=Y1,
                            batchsize=batchsize,
                            learning_rate=learning_rate,
                            n1=n1,
                            nepochs=nepochs)

### END YOUR CODE ###

plt.semilogy(epochs,costs)
plt.show()


# In[ ]:


mtest = 100
Xtest,Ytest = generate_inputs(mtest, func, random=False, vargs=vargs)

# Do the prediction with the trained model

X1, muX, stdevX = normalize(Xtest)
#X1 = Xtest
YpredNorm=predict(W1=params["W1"],W2=params["W2"],X=X1,b1=params["b1"],b2=params["b2"])

#Ypred = YpredNorm
Ypred = inv_normalize(YpredNorm, muX, stdevX)



plt.plot(Xtest[0,:],Ytest[0,:])
plt.plot(Xtest[0,:],Ypred[0,:])


# ### Input Normalisation

# In[ ]:


# Use the normalized inputs and outputs
n1 = 32 # number of hidden units
nepochs = 1000 # number of epochs
batchsize = 10
learning_rate = 0.1


# In[ ]:


X,Y = generate_inputs(m,func,vargs=vargs)

# Input Normalization (average muX and standard deviation stdevX)
X1, muX, stdevX = normalize(X)

# Output Normalization (average muY and standard deviation stdevY)
Y1, muY, stdevY = normalize(Y)




params, epochs, costs=train(X=X1,
                            Y=Y,
                            batchsize=batchsize,
                            learning_rate=learning_rate,
                            n1=n1,
                            nepochs=nepochs)

plt.semilogy(epochs,costs)
plt.show()


# In[ ]:


mtest = 100
Xtest,Ytest = generate_inputs(mtest, func, random=False, vargs=vargs)

# Do the prediction with the trained model

X1, muX, stdevX = normalize(Xtest)
#X1 = Xtest
YpredNorm=predict(W1=params["W1"],W2=params["W2"],X=X1,b1=params["b1"],b2=params["b2"])

#Ypred = YpredNorm
#Ypred = inv_normalize(YpredNorm, muX, stdevX)


plt.plot(Xtest[0,:],Ytest[0,:])
plt.plot(Xtest[0,:],Ypred[0,:])


# ### Output Normalisation

# In[ ]:


X,Y = generate_inputs(m,func,vargs=vargs)

# Input Normalization (average muX and standard deviation stdevX)
X1, muX, stdevX = normalize(X)

# Output Normalization (average muY and standard deviation stdevY)
Y1, muY, stdevY = normalize(Y)



params, epochs, costs=train(X=X,
                            Y=Y1,
                            batchsize=batchsize,
                            learning_rate=learning_rate,
                            n1=n1,
                            nepochs=nepochs)


plt.semilogy(epochs,costs)
plt.show()


# In[ ]:


mtest = 100
Xtest,Ytest = generate_inputs(mtest, func, random=False, vargs=vargs)

# Do the prediction with the trained model

X1, muX, stdevX = normalize(Xtest)
#X1 = Xtest
YpredNorm=predict(W1=params["W1"],W2=params["W2"],X=Xtest,b1=params["b1"],b2=params["b2"])

#Ypred = YpredNorm
Ypred = inv_normalize(YpredNorm, muX, stdevX)


plt.plot(Xtest[0,:],Ytest[0,:])
plt.plot(Xtest[0,:],Ypred[0,:])


# ### No normalisation

# In[ ]:


X,Y = generate_inputs(m,func,vargs=vargs)


params, epochs, costs=train(X=X,
                            Y=Y,
                            batchsize=batchsize,
                            learning_rate=learning_rate,
                            n1=n1,
                            nepochs=nepochs)

plt.semilogy(epochs,costs)
plt.show()


# In[ ]:


mtest = 100
Xtest,Ytest = generate_inputs(mtest, func, random=False, vargs=vargs)

# Do the prediction with the trained model
#X1 = Xtest
Ypred=predict(W1=params["W1"],W2=params["W2"],X=Xtest,b1=params["b1"],b2=params["b2"])


plt.plot(Xtest[0,:],Ytest[0,:])
plt.plot(Xtest[0,:],Ypred[0,:])


# All options are converging towards a minimum, but not at the same rate. According to cost and visual analysis of the fit, the one without any normalization works best. We think it’s still proper learning, also without normalization.

# ### Change Number of hidden Units

# In[ ]:


# Use the normalized inputs and outputs
n1Vec = range(32, 192, 32) # number of hidden units
nepochs = 1000 # number of epochs
batchsize = 10
learning_rate = 0.1
costsVec = []

for n1 in n1Vec:
    X,Y = generate_inputs(m,func,vargs=vargs)

    # Input Normalization (average muX and standard deviation stdevX)
    X1, muX, stdevX = normalize(X)

    # Output Normalization (average muY and standard deviation stdevY)
    Y1, muY, stdevY = normalize(Y)



    params, epochs, costs=train(X=X1,
                                Y=Y1,
                                batchsize=batchsize,
                                learning_rate=learning_rate,
                                n1=n1,
                                nepochs=nepochs)
    
    costsVec.append(costs[-1])


    plt.semilogy(epochs,costs)
    plt.show()


# In[ ]:


plt.plot(n1Vec, costsVec)
plt.show()


# Cost is increasing with higher number of hidden units. Learning curves looks very similar in shape. Curve is getting steeper with higher hidden units.
