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
    return np.array([gradF1(P[0])-lambda0,  gradF2(P[1]) - lambda0, gradF3(P[2])-lambda0])

def lambda_new(P0):
    return np.round((1/3)*(gradF1(P0[0])+gradF2(P0[1])+gradF3(P0[2])),4)

def unit_max_inc_cost(PG):
    return np.argmax(np.array([gradF1(PG),gradF2(PG),gradF3(PG)]))

def unit_min_inc_cost(PG):
    return np.argmin(np.array([gradF1(PG),gradF2(PG),gradF3(PG)]))

def grad3_new(P):
    PG_new= PD - sum(P)
    lambda0 = gradF3(PG_new)
    return np.array([gradF1(P[0])-lambda0,  gradF2(P[1]) - lambda0])



N = 3
delta = 0.000005
PD = 600
P0 = np.array([150,250,250])


PG = P0[:-1]
count = 0;
print(PG)
gradL = grad3_new(PG)
print(gradL)
epsilon = abs(int(10000*(gradL[0])))

while (np.all(gradL > delta) or count < 10):
    count = count + 1
    PG = PG - epsilon*gradL
    gradL = grad3_new(PG)
    P_total = np.append(PG,PD - sum(PG))
    print(P_total, np.sum(P_total))



#F1 = 0.0005PG1^2 + 0.8PG1 + 9 Btu∕h
#F2 = 0.0009PG2^2 + 0.5PG2 + 6 Btu∕h
#F3 = 0.0006PG32 + 0.7PG3 + 8 Btu∕h

