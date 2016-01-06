from scipy.spatial.distance import euclidean
import numpy as np
import random as rndm
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import time

alpha = 1.1
distance_step = 1
neighbours_step = 1
n_dimensions = 9
n_neighbors_for_test = {}
neighbours_needed = []
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
            # print(test_point)
            # print(n_neighbors_for_test[test_point][image_class])
            if((test_point_euc not in n_neighbors_for_test[test_point][image_class] and min_distance > test_point_euc) or min_distance == -1):
                min_distance = test_point_euc
        n_neighbors_for_test[test_point][image_class].append(min_distance)
    return n_neighbors_for_test[test_point][image_class][-1]


def generate_lamdas(training_class_sample_map):
    random_point = [0,0,0,0,0,0,0,0,0]
    sample_count = 0
    for image_class in training_class_sample_map:
        random_point += sum(training_class_sample_map[image_class])
        sample_count += len(training_class_sample_map[image_class])
    random_sample = random_point/float(sample_count)
    # random_sample = training_class_sample_map[rndm.randrange(start=0,stop=len(training_class_sample_map), step=1)][rndm.randrange(start=0,stop=len(training_class_sample_map[0]), step=1)]
    lamda_values = []
    for image_class in training_class_sample_map:
        # print(image_class)
        training_samples = training_class_sample_map[image_class]
        distance = distance_step
        num_samples_per_unit_distance = []
        local_lamdas = []
        ts1 = time.time()
        while True:
            num_samples = 0
            for sample in training_samples:
                distance_from_rand = euc(random_sample,sample)
                if(distance_from_rand < distance):
                    num_samples += 1
            num_samples_per_unit_distance.append(num_samples)
            if(len(num_samples_per_unit_distance) == 1):
                local_lamdas.append(num_samples_per_unit_distance[-1])
            else:
                local_lamdas.append(num_samples_per_unit_distance[-1]-num_samples_per_unit_distance[-2])
            # print("samples per unit distance::" + str(image_class) + "  " + str(sum(num_samples_per_unit_distance)))
            # print("num train samples ::" + str(len(training_samples)))
            if(int(num_samples) == len(training_samples)):
                ts2 = time.time()
                break
            else:
                distance += distance_step

        # print(local_lamdas)
        # print(sum(local_lamdas))
        # print(ts2-ts1)
        lamda_values.append(sum(local_lamdas)/float(len(local_lamdas)))
    print(lamda_values)
    lamda_values_dump = open("data/lamda_values", "wb")
    pickle.dump(lamda_values, lamda_values_dump)

def deltaV(test_point,lamda_values,class_one,class_two):
    if(class_one == class_two):
        return class_one

    lamda_one = lamda_values[class_one]
    lamda_two = lamda_values[class_two]
    if(lamda_two > lamda_one):
        class_plus = class_two
        class_minus = class_one
    else:
        class_plus = class_one
        class_minus = class_two
    lamda_plus = lamda_values[class_plus]
    lamda_minus = lamda_values[class_minus]
    confidence_value = math.log(alpha,2)/(lamda_plus - lamda_minus)
    neighbors = 1
    while True:
        distance_one = nnn_distance_traning_sample_class(test_point,neighbors,class_plus)
        # volume_one = gamma * (distance_one ** (float(n_dimensions)/2))
        distance_two = nnn_distance_traning_sample_class(test_point,neighbors,class_minus)
        # volume_two = gamma * (distance_two ** (float(n_dimensions)/2))

        distance_diff = distance_one - distance_two
        if(distance_diff >= confidence_value):
            neighbours_needed.append(2*neighbors)
            return class_minus
        elif(distance_diff <= (-1 * confidence_value)):
            neighbours_needed.append(2*neighbors)
            return class_plus
        neighbors = neighbors + neighbours_step
        # volume_diff = volume_one - volume_two
        # if(volume_diff >= confidence_value):
        #         return class_one
        # elif(volume_diff <= (-1 * confidence_value)):
        #         return class_two
        # neighbors += neighbours_step



training_class_sample_map = unpickle("data/training_class_sample_map")
test_class_sample_map = unpickle("data/test_class_sample_map")
# generate_lamdas(training_class_sample_map)

# print(lamda_values)
#
# lamda_values =  unpickle("data/lamda_values")
# print(deltaV(test_class_sample_map[1][0],lamda_values,0,1))
lamda_values =  unpickle("data/lamda_values")
correct = 0
wrong = 0
for test_key in range(0,10):
    # flag = 0
    total = 0
    for test_point in test_class_sample_map[test_key]:
        for validation_key in range(0,10):
            if(test_key == validation_key):
                continue
            expected = deltaV(test_point,lamda_values,test_key,validation_key)
            if(test_key == expected):
                correct += 1
                print(str(total) + " correct " + str(test_key) + " vs " + str(validation_key) + " got " + str(expected))
            else:
                wrong += 1
                print(str(total) + " wrong::" + "actual:: " + str(test_key) + "expected ::" + str(expected))

        # flag += 1
        total += 1
        # if(flag == 10):
        #     break
        free_euc(test_point=test_point)
    accuracy = (float(correct))/(correct + wrong)

print(correct)
print(wrong)
print(sum(neighbours_needed)/float(len(neighbours_needed)))
print(accuracy)





# plt.plot(range(20,50),accuracies)
# plt.show()
# print(distance_ones)
# print(distance_twos)
# print(delta_n)
# print(confidence_values)
# print(lamda_ones)
# print(lamda_twos)
# print(count_ones)
# print(count_twos)