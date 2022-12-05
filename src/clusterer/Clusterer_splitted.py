import time
import pickle
import numpy as np
import pandas as pd
from pm4py import read_xes, write_xes
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from statistics import mean, median, stdev

N_CLUSTERS = 100
TRAINING_PATH = "../../data/logs/BPI_2012/BPI_2012_log_eng_ordered_nomicro_rewards_cumulative_durations_training_80.xes"
TESTING_PATH = "../../data/logs/BPI_2012/BPI_2012_log_eng_ordered_nomicro_rewards_cumulative_durations_testing_20.xes"
PATH_CLUSTERED_TRAINING = "../../cluster_data/output_logs/BPI2012_ordered_" + str(N_CLUSTERS) + "_test_fixed_training_80.xes"
PATH_CLUSTERED_TESTING = "../../cluster_data/output_logs/BPI2012_ordered_" + str(N_CLUSTERS) + "_test_fixed_testing_20.xes"


# the denominator in the normalization of the number of event in the trace, could be avg, median or max
event_number_normalization_type = "max"
position_normalization_type = "max"
monitor = False


def create_data_frequency_positional(log, events_set=None, avg_trace_length=None, len_log=None, mms=None, testing=False):
    print("Start encoding dataframe")
    # Computing the set of the different events for the columns of the dataframe
    # dict: for each event (key) in the log it contains the average number of occurence in the traces (value)
    if not testing:
        events_set_list = dict()
        trace_length = []
        len_log = len(log)
        for i, trace in enumerate(log):
            if monitor:
                print("Computing trace stats: trace %s/%s" % (i+1, len_log))
            trace_event_count = dict()
            trace_length += [len(trace)]
            for e in trace:
                event_name = e["concept:name"]
                if event_name not in events_set_list.keys():
                    events_set_list[event_name] = []
                if event_name not in trace_event_count.keys():
                    trace_event_count[event_name] = 1
                else:
                    trace_event_count[event_name] += 1
            for event_name, count in trace_event_count.items():
                events_set_list[event_name] += [count]

        # compute trace stats
        trace_length_aggr = stats(trace_length)
        events_set_aggr = {e: stats(v) for e, v in events_set_list.items()}
        if monitor:
            print('Events statistics:')
            print('trace length: ', trace_length_aggr)
            for item in events_set_aggr.items():
                print(item)
        avg_trace_length = int(trace_length_aggr[position_normalization_type])
        events_set = {e: v[event_number_normalization_type] for e, v in events_set_aggr.items()}
        events_set = {e: max(list(events_set.values())) for e in events_set.keys()}

    last_act_to_cluster = {x: set() for x in events_set}
    data_matrix = list()
    for i, trace in enumerate(log):
        if monitor:
            print("Computing df: trace %s/%s" % (i + 1, len_log))
        # Initializing a dictionary to store the number of occurrences for each event
        events_dict = {e: {'count': 0, 'last_position': 0} for e in events_set}
        j = 0.0
        for j, event in enumerate(trace):
            # Updating the dictionary
            if event["concept:name"] in events_dict.keys():
                events_dict[event["concept:name"]]['count'] += 1
                events_dict[event["concept:name"]]['last_position'] = j + 1
            # al posto di event_set[e] ci dobbiamo mettere o la media o il max delle tracce del log
            to_add = [i, j] + [float(d['count'] / events_set[e]) for e, d in events_dict.items()] + [
                float(d['last_position'] / avg_trace_length) for e, d in events_dict.items()]
            if event["concept:name"] == "END":
                to_add.append(event["kpi:reward"])
            else:
                to_add.append(0.0)
            data_matrix.append(to_add)
    data = pd.DataFrame(data_matrix)
    # Initializing a MinMaxScaler for the continuous column: reward
    if not testing:
        mms = MinMaxScaler(feature_range=(0, 1))
        # Transforming the two continuous columns
        data[len(data.transpose()) - 1] = mms.fit_transform(data[[len(data.transpose()) - 1]])
    else:
        data[len(data.transpose()) - 1] = mms.transform(data[[len(data.transpose()) - 1]])

    if testing:
        return data
    else:
        return data, mms, events_set, avg_trace_length, len_log, last_act_to_cluster


def stats(list):
   stats = {'max': max(list), 'avg': mean(list), 'median': median(list), 'stdev': stdev(list),
                         'Q3': np.percentile(list, 75), 'D9': np.percentile(list, 90)}
   return stats


def cluster(df, log, mms, last_a_to_cluster):
    print("Number of items: " + str(len(df)))
    np.nan_to_num(df)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(df[[x for x in range(2, len(df.transpose()))]])
    print("Finished Clustering, now starting exporting results")
    df.apply(iterTrainingDf, args=(log, last_a_to_cluster, mms, kmeans), axis=1)
    write_xes(log, PATH_CLUSTERED_TRAINING)
    print("Finished writing training log")
    filename = '../../cluster_data_old/models_encoders/kmeans_model_new.sav'
    pickle.dump(kmeans, open(filename, 'wb'), protocol=4)

    return kmeans


def fit_test_data(df, log, mms, kmean, last_a_to_cluster):
    print("Number of items: " + str(len(df)))
    np.nan_to_num(df)

    labels = kmean.transform(df[[x for x in range(2, len(df.transpose()))]])
    print("Finished Clustering, now starting exporting results")
    df.apply(iterTestingDf, args=(log, last_a_to_cluster, mms, kmean, labels), axis=1)

    write_xes(log, PATH_CLUSTERED_TESTING)
    print("Finished writing testing log")


def iterTrainingDf(row, log, last_a_to_cluster, mms, kmeans):
    event = log[int(row[0])][int(row[1])]
    if event["concept:name"] == "A_PREACCEPTED":
        pass
    label = kmeans.labels_[row.name]
    event["cluster:prefix"] = label
    event["stato"] = event["concept:name"] + " | " + str(label)
    event["cluster:reward"] = mms.inverse_transform([[kmeans.cluster_centers_[label][-1]]])[0][0]
    last_a_to_cluster[event["concept:name"]].add(label)


def iterTestingDf(row, log, last_a_to_cluster, mms, kmean, labels):
    event = log[int(row[0])][int(row[1])]
    best_cluster_value, best_cluster = min([(labels[row.name][x], x) for x in last_a_to_cluster[event["concept:name"]]],
                                           key=lambda x: x[0])
    event["cluster:prefix"] = best_cluster
    event["stato"] = event["concept:name"] + " | " + str(best_cluster)
    event["cluster:reward"] = mms.inverse_transform([[kmean.cluster_centers_[best_cluster][-1]]])[0][0]


if __name__ == "__main__":
    t1 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t1)))
    training_log = read_xes(TRAINING_PATH)
    training_data, training_mms, eventsset, avgtracelength, lenlog, last_act_to_cluster = create_data_frequency_positional(training_log)
    kmeans = cluster(training_data, training_log, training_mms, last_act_to_cluster)
    testing_log = read_xes(TESTING_PATH)
    testing_data = create_data_frequency_positional(testing_log, eventsset, avgtracelength, lenlog, training_mms, True)
    fit_test_data(testing_data, testing_log, training_mms, kmeans, last_act_to_cluster)
    # cluster(pickle.load(open('../../cluster_data/models_encoders/df_V2.sav', 'rb')), read_xes(PATH_COMPUTE_CLUSTER))
    t2 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t2)))
    print(t2 - t1)
