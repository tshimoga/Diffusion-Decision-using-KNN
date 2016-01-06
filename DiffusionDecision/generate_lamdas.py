import time
from scipy.spatial.distance import euclidean
import pickle
import numpy as np

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


training_class_sample_map = unpickle("data/training_class_sample_map")
test_class_sample_map = unpickle("data/test_class_sample_map")

random_point = [0,0,0,0,0,0,0,0,0]
sample_count = 0
for image_class in training_class_sample_map:
    random_point += sum(training_class_sample_map[image_class])
    sample_count += len(training_class_sample_map[image_class])
random_sample = random_point/float(sample_count)
distance_step = 0.1
lamda_values = []
for image_class in training_class_sample_map:
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
        if(int(num_samples) == len(training_samples)):
            ts2 = time.time()
            break
        else:
            distance += distance_step
    lamda_values.append(sum(local_lamdas)/float(len(local_lamdas)))
print(lamda_values)
lamda_values_dump = open("data/lamda_values", "wb")
pickle.dump(lamda_values, lamda_values_dump)