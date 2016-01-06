from scipy.spatial.distance import euclidean
import numpy as np
import math
from decimal import Decimal

alpha = 1.1
distance_step = 1
neighbours_step = 1
n_dimensions = 9
n_neighbors_for_test = {}
neighbours_needed = []
confidence_value = 0.65
a = 1
b = 1
gamma = (math.pi ** (float(n_dimensions)/2))/ math.factorial(5)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

euc_dist = {}


def euc(test_point, point):
    test_point_str = test_point
    test_point_str = ''.join(test_point_str.astype(np.str))
    point_str = point
    point_str = '.'.join(point_str.astype(np.str))
    if(test_point_str in euc_dist and point_str in euc_dist[test_point_str]):
        return euc_dist[test_point_str][point_str]
    else:
        if(test_point_str not in euc_dist):
            euc_dist[test_point_str] = {}
        euc_dist[test_point_str][point_str] = euclidean(test_point,point)
        return euc_dist[test_point_str][point_str]

def free_euc(test_point):
    test_point = ''.join(test_point.astype(np.str))
    euc_dist.pop(test_point)
    n_neighbors_for_test.pop(test_point)

def nnn_distance_traning_sample_class(test_point,n, image_class):
    test_point_eucs = []
    class_traning_sample = training_class_sample_map[image_class]
    for training_point in class_traning_sample:
        test_point_eucs.append(euc(test_point,training_point))
    test_point = ''.join(test_point.astype(np.str))
    if(test_point not in n_neighbors_for_test):
        n_neighbors_for_test[test_point] = {}
    if(image_class not in n_neighbors_for_test[test_point]):
            n_neighbors_for_test[test_point][image_class] = []
    for i in range(0,n):
        min_distance = -1
        for test_point_euc in test_point_eucs:
            if((test_point_euc not in n_neighbors_for_test[test_point][image_class] and min_distance > test_point_euc) or min_distance == -1):
                min_distance = test_point_euc
        n_neighbors_for_test[test_point][image_class].append(min_distance)
    return n_neighbors_for_test[test_point][image_class][n-1]

def ncr(n,r):
    f = math.factorial
    return f(n)/(f(n-r) * f(r))

def deltaPV(test_point,class_one,class_two):
    neighbors = 1
    probablilities = []
    while True:
        distance_one = nnn_distance_traning_sample_class(test_point,neighbors,class_one)
        volume_one = Decimal(gamma * (distance_one ** (float(n_dimensions)/2)))
        distance_two = nnn_distance_traning_sample_class(test_point,neighbors,class_two)
        volume_two = Decimal(gamma * (distance_two ** (float(n_dimensions)/2)))

        u_one = Decimal(neighbors * volume_one)
        u_two = Decimal(neighbors * volume_two)

        k = (2 * neighbors) + 2 * a - 1
        p = Decimal(0)
        for m in range(0,neighbors+1):
            p += Decimal(ncr(k,m) * ((u_one + b) ** m) * ((u_two + b) ** (k-m)))
        p /= Decimal((u_one + u_two + Decimal(2.0) * b) ** k)
        if(p >= confidence_value):
            neighbours_needed.append(neighbors * 2)
            return class_one
        elif(p <= 1-confidence_value):
            neighbours_needed.append(neighbors * 2)
            return class_two
        neighbors += neighbours_step
        # print(p)
        probablilities.append(p)
        if(neighbors >= 50):
            minim = min(probablilities)
            maxim = 1-max(probablilities)
            if(minim > maxim ):
                neighbours_needed.append(neighbors * 2)
                return class_one
            else:
                neighbours_needed.append(neighbors * 2)
                return class_two





training_class_sample_map = unpickle("data/training_class_sample_map")
test_class_sample_map = unpickle("data/test_class_sample_map")

# print(deltaV(test_class_sample_map[1][0],0,1))
correct = 0
wrong = 0
for test_key in range(0,10):
    flag = 0
    total = 0
    for test_point in test_class_sample_map[test_key]:
        for validation_key in range(0,10):
            if(test_key == validation_key):
                continue
            expected = deltaPV(test_point,test_key,validation_key)
            if(test_key == expected):
                correct += 1
                print(str(total) + " correct " + str(test_key) + " vs " + str(validation_key) + " got " + str(expected))
            else:
                wrong += 1
                print(str(total) + " wrong::" + "actual:: " + str(test_key) + "expected ::" + str(expected))
        flag += 1
        total += 1
        if(flag == 50):
            break
        free_euc(test_point=test_point)
    accuracy = (float(correct))/(correct + wrong)

print(correct)
print(wrong)
print("confidence_value : " + str(confidence_value))
print(" Average Neighbors used" + str(sum(neighbours_needed)/float(len(neighbours_needed))))
print("Accuracy :" + str(accuracy))