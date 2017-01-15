import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
 
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
print("Input dataset")
print(X)
 
# output dataset           
y = np.array([[0,0,1,1]]).T
print("Output dataset")
print(y)
 
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
 
# initialize weights randomly with mean 0
print("Initial synapse weights (randomized): ")
syn0 = 2*np.random.random((3,1)) - 1
print(syn0)

for iter in range(1):
    print("ITERATION: ",iter)
    
    # forward propagation
    L0 = X
    print("First layer (L0): ",L0)
    print("Synapse value: ",syn0)
    dot_pr1 = np.dot(L0,syn0)
    print("Dot product: ",dot_pr1)
    L1 = nonlin(dot_pr1)
    print("Second layer (L1): ",L1)
     
    # how much did we miss?
    L1_error = y - L1
    print("Miss (y - L1): ",L1_error)
     
    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    deriv1 = nonlin(L1,True)
    print("Slope of sigmoid at L1 values: ",deriv1)
    L1_delta = L1_error*deriv1
    print("L1 delta: ",L1_delta)

    # update weights
    print("Updating weights: ",syn0)
    dot_pr2 = np.dot(L0.T,L1_delta)
    print("Dot product of L0.T and L1_delta: ",dot_pr2)
    syn0 += dot_pr2
    print("Updated synapse weights: ",syn0)
 
print("Output After Training:")
print(L1)
