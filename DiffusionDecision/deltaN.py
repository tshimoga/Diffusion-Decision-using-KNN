from scipy.spatial.distance import euclidean
import numpy as np
import math
import sys

alpha = 7
distance_step = 0.1
neighbours_needed = []
n_neighbors_for_test = []
neighbor_limit = 400

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

def deltaN(training_class_sample_map,test_point,lamda_values,class_one,class_two):
    lamda_one = lamda_values[class_one]
    lamda_two = lamda_values[class_two]
    if(lamda_one < lamda_two):
        temp = class_one
        class_one = class_two
        class_two = temp
        temp = lamda_one
        lamda_one = lamda_two
        lamda_two = temp
    training_class_sample_one = training_class_sample_map[class_one]
    training_class_sample_two = training_class_sample_map[class_two]
    confidence_value = math.log10(alpha)/math.log10(lamda_one/lamda_two)
    distance = distance_step
    count_diff_sum = 0
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
        count_diff = count_one - count_two
        if(count_diff >= confidence_value):
            neighbours_needed.append(count_one + count_two)
            return class_one
        elif(count_diff <= (-1 * confidence_value)):
            neighbours_needed.append(count_one + count_two)
            return class_two
        if(count_one == 0 and count_two == 0):
            distance = min(nnn_distance_traning_sample_class(test_point,class_one),nnn_distance_traning_sample_class(test_point,class_two)) + distance_step
            continue
        distance += distance_step
        if(count_one + count_two >= neighbor_limit):
            if(count_diff_sum < 0):
                return class_two
            else:
                return class_one
        else:
            count_diff_sum += count_diff




training_class_sample_map = unpickle("data/training_class_sample_map")
test_class_sample_map = unpickle("data/test_class_sample_map")
lamda_values =  unpickle("data/lamda_values")

correct = 0
wrong = 0
for test_key in range(0,10):
    for test_point in test_class_sample_map[test_key]:
        for validation_key in range(0,10):
            if(test_key == validation_key):
                continue
            expected = deltaN(training_class_sample_map,test_point,lamda_values,test_key,validation_key)
            if(test_key == expected):
                correct += 1
            else:
                wrong += 1
        free_euc(test_point=test_point)
    accuracy = (float(correct))/(correct + wrong)

print("alpha" + str(alpha))
print("Neigbors used:" + str(sum(neighbours_needed)/float(len(neighbours_needed))))
print("accuracy" + str(accuracy))
