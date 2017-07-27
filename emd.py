import numpy as np
from scipy.optimize import linprog
import math

def EMD(amp1, f1, amp2, f2, noiseCancellationParameter=0.05):
    W1 = [] #store the amplitudes as weights
    X = []; #store the freq. as the data points which need to be moved
    for j in range(amp1.shape[0]):
       if(amp1[j] <= noiseCancellationParameter):
           continue
       else:
           W1.append(amp1[j] - noiseCancellationParameter)
           X.append(f1[j])

    #Now truncate the array to required size
    '''index = find(W1 == 0,1,'first');
    
    %this step is important because, sometimes, element zero may not be
    %present in the weigths
    if(~isempty(index))
        W1 = W1(1:index-1);
        X = X(1:index-1);
    end'''
    W1 = np.array(W1)
    X = np.array(X)
   
    W2 = [];
    Y = [];

    for j in range(amp2.shape[0]):
       if(amp2[j] <= noiseCancellationParameter):
           continue
       else:
           W2.append(amp2[j] - noiseCancellationParameter)
           Y.append(f2[j])

    #Now truncate the array to required size
    '''index = find(W2 == 0,1,'first');
    
    if(~isempty(index))
        W2 = W2(1:index-1);
    Y = Y(1:index-1);
    end'''
    W2 = np.array(W2)
    Y = np.array(Y)
    
    #Calculate the distance matrix
    #We will be using the euclidean distance as the ground distance
    #Since, what we have is 1-dimensional, we get simply |X(i) - Y(j)|
    m = X.shape[0];
    n = Y.shape[0];
    D = np.zeros(shape=(m,n))
    for i in range(m):
        for j in range(n):
            D[i][j] = np.abs(X[i] - Y[j])

    D = D.T
    D = D.reshape(-1)

    #inequality constraints
    A1 = np.zeros(shape=(m, m * n))
    A2 = np.zeros(shape=(n, m * n))
    for i in range(m):
        for j in range(n):
            k = j + (i - 1) * n;
            A1[i][k] = 1;
            A2[j][k] = 1;

    A = np.concatenate((A1, A2), axis=0)
    b = np.concatenate((W1, W2), axis=0)
    
    #equality constraints
    Aeq = np.ones(shape=(m + n, m * n))
    beq = np.ones(shape=(m + n, 1)) * min(sum(W1), sum(W2));
    
    #lower bound
    lb = np.zeros(shape=(m*n,1))
    t = m*n;
    #linear programming
    #case where m = n = 0. Note that m and n are positive integers
    if(m+n == 0):
        emd = 0
    #this means either m or n is 0 but not both in which case emd is Inf
    elif m + n > 0 and m*n==0:
        emd = math.inf
    else:
        if(np.all(D == 0)):
            emd = 0;
        else:
            '''cvx_begin quiet
                variable x(t);
                minimize (D'*x);
                subject to
                    A*x <= b;
                    Aeq*x == beq;
                    x >= lb;
            cvx_end'''
            result = linprog(D.T,A,b,Aeq,beq)
            print(result.x)
            '''emd = D'*x/sum(x);
            if(isnan(emd)):
                emd = 0;'''


if __name__ == '__main__':
    amp1 = np.array([0.0, 1.0])
    amp2 = np.array([5.0, 3.0])
    f1 = np.array([1,2])
    f2 = np.array([3,4])
    EMD(amp1,f1,amp2,f2,0)


