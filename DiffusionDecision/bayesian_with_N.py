from scipy.spatial.distance import euclidean
import numpy as np
import math
import sys
from decimal import Decimal

alpha = 1.1
distance_step = 0.1
neighbours_needed = []
n_neighbors_for_test = []
confidence_value = 0.65

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
    test_point_str = test_point
    test_point_str = ''.join(test_point_str.astype(np.str))
    euc_dist.pop(test_point_str)

def nnn_distance_traning_sample_class(test_point,image_class):
    class_traning_sample = training_class_sample_map[image_class]
    min_distance = sys.maxint
    for training_point in class_traning_sample:
        euclid = euc(test_point,training_point)
        if(min_distance > euclid):
            min_distance = euclid
    return min_distance
def ncr(n,r):
    f = math.factorial
    return f(n)/(f(n-r) * f(r))

def deltaPN(training_class_sample_map,test_point,class_one,class_two):
    training_class_sample_one = training_class_sample_map[class_one]
    training_class_sample_two = training_class_sample_map[class_two]
    distance = distance_step
    while True:
        count_one = 0
        for point in training_class_sample_one:
            point_euc = euc(test_point, point)
            if(point_euc < distance):
                count_one+=1
        count_two = 0
        for point in training_class_sample_two:
            point_euc = euc(test_point, point)
            if(point_euc < distance):
                count_two+=1
        k = count_one + count_two + 1
        p = 0
        if(k > 1000):
            p = Decimal(0.0)
            for m in range(0,count_one+1):
                p += Decimal(ncr(k,m))
            denom = 2 ** k
            p = Decimal(p / denom)
        else:
            p = 0.0
            for m in range(0,count_one+1):
                p += ncr(k,m)
            denom = 2 ** k
            p = float(p) / denom
        if(p >= confidence_value):
            neighbours_needed.append(count_one + count_two)
            return class_one
        elif(p <= 1-confidence_value):
            neighbours_needed.append(count_one + count_two)
            return class_two
        if(count_one == 0 and count_two == 0):
            distance = min(nnn_distance_traning_sample_class(test_point,class_one),nnn_distance_traning_sample_class(test_point,class_two)) + distance_step
            continue
        distance += distance_step



training_class_sample_map = unpickle("data/training_class_sample_map")
test_class_sample_map = unpickle("data/test_class_sample_map")

correct = 0
wrong = 0
for test_key in range(0,10):
    for test_point in test_class_sample_map[test_key]:
        for validation_key in range(0,10):
            if(test_key == validation_key):
                continue
            expected = deltaPN(training_class_sample_map,test_point,test_key,validation_key)
            if(test_key == expected):
                correct += 1
            else:
                wrong += 1
        free_euc(test_point=test_point)
    accuracy = (float(correct))/(correct + wrong)

print("confidence_value:" + str(confidence_value))
print("Neighbors Used:" + str(sum(neighbours_needed)/float(len(neighbours_needed))))
print("accuracy" + str(accuracy))
