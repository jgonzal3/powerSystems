import numpy as np

def gradF1(x):
    return 2*0.0005*x+0.8

def gradF2(x):
    return 2*0.0009*x+0.5

def gradF3(x):
    return 2*0.0006*x+0.7

def F1(x):
    return 0.0005**x+0.8*x + 9

def F2(x):
    return 0.0009**x+0.5*x + 6

def F3(x):
    return 0.0006**x+0.7*x + 8

def grad_new(P,lambda0):
    return np.array([gradF1(P[0])-lambda0,  gradF2(P[1]) - lambda0, gradF3(P[2])-lambda0, PD-np.sum(P)])

def lambda_new(P0):
    return np.round((1/3)*(gradF1(P0[0])+gradF2(P0[1])+gradF3(P0[2])),2)

N = 3
delta = 0.00000005
PD = 600
P0 = np.array([144,248,206])
lambda0 = lambda_new(P0)
print(lambda0)

PG = np.append(P0, lambda0)
count = 0;
gradL = grad_new(P0,lambda0)
epsilon = abs(int(10000*(gradL[0])))


while (np.all(gradL[:-1] > delta) or count < 1000):
    count = count + 1
    print (count)
    PG = PG - epsilon*gradL
    lambda0 = lambda_new(PG[:-1])
    gradL = grad_new(PG[:-1],lambda0)
    print(PG)



#F1 = 0.0005PG1^2 + 0.8PG1 + 9 Btu∕h
#F2 = 0.0009PG2^2 + 0.5PG2 + 6 Btu∕h
#F3 = 0.0006PG32 + 0.7PG3 + 8 Btu∕h

