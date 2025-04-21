def f(N,x):
    if (N==1):
        res = 0.5 * 0.77 * x**2 + 23.5 * x
    if (N==2):
        res = 0.5 * 1.60 * x**2 + 26.5 * x
    if (N==3):
        res = 0.5 * 2 * x**2 + 30.0 * x
    if (N==4):
        res = 0.5 * 2.5 * x**2 + 32.0 * x
    return res


def F_N(N,x):
    if N == 1:
        return f(N, x)
    
    min_value = 99999999
    #print(N)
    for y in range(x + 1):
        min_value = (min(min_value, f(N,y) + F_N(N-1, x - y)))
    
    return min_value

sol = {}
for n in range(1,5):
    for load in range(1,49):
        result = F_N(n,load)
        sol[(n,load)] = result

for l in range(1,49):
    my_list = [sol[(1,l)],sol[(2,l)],sol[(3,l)],sol[(4,l)]]
    min_value = min(my_list)
    position = my_list.index(min_value)
    print (f"{l} MW of load is satisfied with {position+1} generators")


