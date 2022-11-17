import time
import pickle
import numpy as np
import pandas as pd
from pm4py import read_xes, write_xes
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

N_CLUSTERS = 50
TRAINING_PATH = "../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations_ordered_V2_training_80.xes"
TESTING_PATH = "../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations_ordered_V2_testing_20.xes"
PATH_CLUSTERED_TRAINING = "../../cluster_data/output_logs/BPI2012_ordered_" + N_CLUSTERS + "_positional_cumulative_squashed_V2_training_80.xes"
PATH_CLUSTERED_TESTING = "../../cluster_data/output_logs/BPI2012_ordered_" + N_CLUSTERS + "_positional_cumulative_squashed_V2_testing_20.xes"

def cluster(df, log, mms):
    print("Number of items: " + str(len(df)))
    np.nan_to_num(df)
    last_a_to_cluster = dict()
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(df[[x for x in range(2, len(df.transpose()))]])
    print("Finished Clustering, now starting exporting results")
    for row in df.transpose():
        event = log[int(df.transpose()[row][0])][int(df.transpose()[row][1])]
        event["cluster:prefix"] = kmeans.labels_[row]
        event["stato"] = log[int(df.transpose()[row][0])][int(df.transpose()[row][1])]["concept:name"] + " | " + str(kmeans.labels_[row])
        event["cluster:reward"] = mms.inverse_transform([[kmeans.cluster_centers_[kmeans.labels_[row]][-1]]])[0][0]
        if event["concept:name"] not in last_a_to_cluster.keys():
            last_a_to_cluster[event["concept:name"]] = set()
        last_a_to_cluster[event["concept:name"]].add(kmeans.labels_[row])

    write_xes(log, PATH_CLUSTERED_TRAINING)
    print("Finished writing training log")
    #filename = '../../cluster_data_old/models_encoders/kmeansV2_100.sav'
    #pickle.dump(kmeans, open(filename, 'wb'), protocol=4)

    return kmeans, last_a_to_cluster

def create_data_frequency_positional(log):
    print("Start encoding dataframe")
    # Computing the set of the different events for the columns of the dataframe
    events_set = dict()
    avg_trace_length = 0
    for trace in log:
        avg_trace_length += len(trace)
        for e in trace:
            if e["concept:name"] not in events_set.keys():
                events_set[e["concept:name"]] = 1
            else:
                events_set[e["concept:name"]] += 1

    avg_trace_length = int(avg_trace_length / len(log))
    data_matrix = list()

    i = 0.0
    for trace in log:
        # Initializing a dictionary to store the number of occurrences for each event
        events_dict = {e: {'count': 0, 'last_position': 0} for e in events_set}
        j = 0.0
        for event in trace:
            # Updating the dictionary
            if event["concept:name"] in events_dict.keys():
                events_dict[event["concept:name"]]['count'] += 1
                events_dict[event["concept:name"]]['last_position'] = j + 1
            to_add = [i, j] + [float(d['count'] / events_set[e]) for e, d in events_dict.items()] + [
                float(d['last_position'] / avg_trace_length) for e, d in events_dict.items()]
            to_add.append(event["kpi:reward"])
            data_matrix.append(to_add)
            j += 1
        i += 1
    data = pd.DataFrame(data_matrix)
    # Initializing a MinMaxScaler for each continuous column: reward and amount
    mms = MinMaxScaler(feature_range=(-1, 1))
    # Transforming the two continuous columns
    data[len(data.transpose()) - 1] = mms.fit_transform(data[[len(data.transpose()) - 1]])

    return data, mms


def fit_test_data(df, log, mms, kmean, last_a_to_c):
    print("Number of items: " + str(len(df)))
    np.nan_to_num(df)

    labels = kmean.transform(df[[x for x in range(2, len(df.transpose()))]])
    print("Finished Clustering, now starting exporting results")
    for row in df.transpose():
        event = log[int(df.transpose()[row][0])][int(df.transpose()[row][1])]
        best_cluster_value, best_cluster = max([(labels[row][x], x) for x in last_a_to_cluster[event["concept:name"]]], key=lambda x: x[0])
        event["cluster:prefix"] = best_cluster
        event["stato"] = log[int(df.transpose()[row][0])][int(df.transpose()[row][1])]["concept:name"] + " | " + str(best_cluster)
        event["cluster:reward"] = mms.inverse_transform([[kmean.cluster_centers_[best_cluster][-1]]])[0][0]

    write_xes(log, PATH_CLUSTERED_TESTING)
    print("Finished writing testing log")
    #filename = '../../cluster_data_old/models_encoders/kmeansV2_100.sav'
    #pickle.dump(kmeans, open(filename, 'wb'), protocol=4)


if __name__ == "__main__":
    t1 = time.time()
    training_log = read_xes(TRAINING_PATH)
    training_data, training_mms = create_data_frequency_positional(training_log)
    kmeans, last_a_to_cluster = cluster(training_data, training_log, training_mms)
    testing_log = read_xes(TESTING_PATH)
    testing_data, testing_mms = create_data_frequency_positional(testing_log)
    fit_test_data(testing_data, testing_log, testing_mms, kmeans, last_a_to_cluster)
    # cluster(pickle.load(open('../../cluster_data/models_encoders/df_V2.sav', 'rb')), read_xes(PATH_COMPUTE_CLUSTER))
    t2 = time.time()
    print(t2 - t1)
