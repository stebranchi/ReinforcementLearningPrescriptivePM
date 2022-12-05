import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pm4py import read_xes, write_xes
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from statistics import mean, median, stdev


TRAINING_PATH = "../../data/logs/BPI_2012/BPI_2012_log_eng_ordered_nomicro_rewards_cumulative_durations_training_80.xes"
# TESTING_PATH = "../input_data/clusters/BPI_2012/50_clusters/log/BPI_2012_log_eng_rewards_cumulative_durations_testing_20.xes"
# PATH_CLUSTERED_TRAINING = "../input_data/clusters/BPI_2012/50_clusters/log/BPI2012_log_eng_positional_cumulative_squashed_training_80.xes"
# PATH_CLUSTERED_TESTING = "../input_data/clusters/BPI_2012/50_clusters/log/BPI2012_log_eng_positional_cumulative_squashed_testing_20.xes"

#TRAINING_PATH = "../input_data/clusters/BPI_2012/log_base/BPI_2012_log_eng_ordered_nomicro_rewards_cumulative_durations_training_80.xes"

# the denominator in the normalization of the number of event in the trace, could be avg, median or max
monitor = True
all_minmax = False
clean_encoding = False
last_activity_importance = 0
kmin = 10
kmax = 100
step = 10
iterations = 1

def create_data_frequency_positional(log):
    print("Start encoding dataframe")
    # Computing the set of the different events for the columns of the dataframe
    # dict: for each event (key) in the log it contains the average number of occurence in the traces (value)
   # get event set
    event_set = get_event_set(log, event_label="concept:name")
    #initialize event frequency and position and trace length lists
    event_frequency_list = {e: [] for e in event_set}
    event_first_position_list = {e: [] for e in event_set}
    trace_length = []
    len_log = len(log)
    # extract event frequency and position for trace
    for i, trace in enumerate(log):
        if monitor:
            print("Computing trace stats: trace %s/%s" % (i + 1, len_log))
        trace_event_frequency = {e: 0 for e in event_set}
        trace_event_first_position = {e: 0 for e in event_set}
        trace_length += [len(trace)]
        for j, e in enumerate(trace):
            event_name = e["concept:name"]
            # add 1 count to the frequency
            trace_event_frequency[event_name] += 1
            # add the position of the first encounter
            if trace_event_first_position[event_name] == 0:
                trace_event_first_position[event_name] = j+1
        for event_name, count in trace_event_frequency.items():
            event_frequency_list[event_name] += [count]
        for event_name, position in trace_event_first_position.items():
            event_first_position_list[event_name] += [position]

    # compute trace stats
    trace_length_aggr = stats(trace_length)
    events_frequency_aggr = {e: stats(v, zeros=True) for e, v in event_frequency_list.items()}
    event_first_position_aggr = {e: stats(v, zeros=False) for e, v in event_first_position_list.items()}
    if monitor:
        print('Events statistics:')
        print('trace length: ', trace_length_aggr)
        print('Frequencies:')
        for item in events_frequency_aggr.items():
            print(item)
        print('Positions:')
        for item in event_first_position_aggr.items():
            print(item)
    avg_trace_length = int(trace_length_aggr['1'])
    events_frequency = {e: v['1'] for e, v in events_frequency_aggr.items()}

    if clean_encoding:
        event_to_remove_freq = [e for e, v in events_frequency_aggr.items() if v['min'] == v['max']]
        event_to_remove_pos = [e for e, v in event_first_position_aggr.items() if v['min'] == v['max']]
        if monitor:
            print('Cleaning the encoding:')
            print('event_to_remove_freq:', event_to_remove_freq)
            print('event_to_remove_pos:', event_to_remove_pos)
        # for e in event_to_remove_freq:
        #     events_frequency.pop(e, None)
    else:
        event_to_remove_freq = []
        event_to_remove_pos = []

    events_frequency = {e: max(list(events_frequency.values())) for e in events_frequency.keys()}

    data_matrix = list()

    for i, trace in enumerate(log):
        if monitor:
            print("Computing df: trace %s/%s" % (i + 1, len_log))
        # Initializing a dictionary to store the number of occurrences for each event
        events_dict = {e: {'count': 0, 'last_position': 0} for e in event_set}
        for j, event in enumerate(trace):
            event_name = event["concept:name"]
            # Updating the dictionary
            if event_name in events_dict.keys():
                events_dict[event_name]['count'] += 1
                if events_dict[event_name]['last_position'] == 0:
                    events_dict[event_name]['last_position'] = j + 1
            if all_minmax:
                to_add = [i, j] + [d['count'] for e, d in events_dict.items() if e not in event_to_remove_freq] + \
                         [d['last_position'] for e, d in events_dict.items() if e not in event_to_remove_pos]
            elif last_activity_importance == 0:
                to_add = [i, j] + [float(d['count'] / events_frequency[e]) for e, d in events_dict.items() if e not in event_to_remove_freq] + \
                         [float(d['last_position'] / avg_trace_length) for e, d in events_dict.items() if e not in event_to_remove_pos]
            else:
                to_add = [i, j] + [float(d['count'] / events_frequency[e]) for e, d in events_dict.items() if e not in event_to_remove_freq] + \
                         [float(d['last_position'] / avg_trace_length) for e, d in events_dict.items() if e not in event_to_remove_pos] + \
                         [last_activity_importance if e == event_name else 0 for e in events_dict.keys()]
            reward = event["kpi:reward"] if event["concept:name"] == "END" else 0.0
            to_add.append(reward)
            data_matrix.append(to_add)
    data = pd.DataFrame(data_matrix)
    # Initializing a MinMaxScaler for the continuous column: reward
    mms = MinMaxScaler(feature_range=(-1, 1))
    if all_minmax:
        # Transforming all the columns but the first 2
        data[list(range(2,len(data.transpose())))] = mms.fit_transform(data[list(range(2,len(data.transpose())))])
    else:
        # Transforming the two continuous columns (reward)
        data[len(data.transpose()) - 1] = mms.fit_transform(data[[len(data.transpose()) - 1]])

    return data, mms


def get_event_set(log, event_label):
    event_set = set()
    for t in log:
        for e in t:
            event_set.add(e[event_label])
    return event_set


def stats(list, zeros=True):
    if not zeros:
        list = [x for x in list if x != 0]
    stats = {'min': min(list), 'max': max(list), 'avg': mean(list), 'median': median(list), 'stdev': stdev(list),
             'Q3': np.percentile(list, 75), 'D9': np.percentile(list, 90), '1': 1}
    return stats

def compute_silhoutte(row, kmeans, silhouette_scores, k):
    # x = row[1].to_numpy()
    x = row.to_numpy()
    distances = np.sort(kmeans.transform([x]))
    a, b = distances[0, :2]
    silhouette_scores[k] += (b - a) / max(a, b)

def Silhouette_search(df, log, mms):
    print("Number of items: " + str(len(df)))
    np.nan_to_num(df)
    silhouette_scores = dict()
    scaled_features = df[[x for x in range(2, len(df.transpose()))]]
    print('Start searching k space:')
    for k in range(kmin, kmax+1, step):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_features)
        silhouette_scores[k] = 0
        for i in range(iterations):
            print('k: %s, iteration: %s/%s' % (k, i+1, iterations))
            # for index, row in scaled_features.iterrows():
            #     x = row.to_numpy()
            #     distances = np.sort(kmeans.transform([x]))
            #     a, b = distances[0, :2]
            #     silhouette_scores[k] += (b - a) / max(a, b)
            scaled_features.apply(compute_silhoutte, args=(kmeans, silhouette_scores, k), axis=1)
        silhouette_scores[k] = silhouette_scores[k] / len(scaled_features) / iterations

    print('Silhouette:', silhouette_scores)
    s_list = silhouette_scores.items()
    s_list = sorted(s_list)
    x, y = zip(*s_list)
    # plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c='black', s=50)
    plt.plot(x, y)
    plt.savefig('../../cluster_data/cluster_k_analysis_' + time.strftime("%Y%m%d%H%M%S", time.gmtime(time.time())) + '.png')


# def fit_test_data(df, log, mms, kmean):
#     print("Number of items: " + str(len(df)))
#     np.nan_to_num(df)
#
#     labels = kmean.predict(df[[x for x in range(2, len(df.transpose()))]])
#     print("Finished Clustering, now starting exporting results")
#     for row in df.transpose():
#         log[int(df.transpose()[row][0])][int(df.transpose()[row][1])]["cluster:prefix"] = labels[row]
#         log[int(df.transpose()[row][0])][int(df.transpose()[row][1])]["stato"] = log[int(df.transpose()[row][0])][int(df.transpose()[row][1])]["concept:name"] + " | " + str(labels[row])
#         log[int(df.transpose()[row][0])][int(df.transpose()[row][1])]["cluster:reward"] = mms.inverse_transform([[kmean.cluster_centers_[labels[row]][-1]]])[0][0]
#
#     write_xes(log, PATH_CLUSTERED_TESTING)
#     print("Finished writing testing log")
#     #filename = '../../cluster_data_old/models_encoders/kmeansV2_100.sav'
#     #pickle.dump(kmeans, open(filename, 'wb'), protocol=4)


if __name__ == "__main__":
    t1 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t1)))
    training_log = read_xes(TRAINING_PATH)
    training_data, training_mms = create_data_frequency_positional(training_log)
    Silhouette_search(training_data, training_log, training_mms)
    # testing_log = read_xes(TESTING_PATH)
    # testing_data, testing_mms = create_data_frequency_positional(testing_log)
    # fit_test_data(testing_data, testing_log, testing_mms, kmeans)
    # # cluster(pickle.load(open('../../cluster_data/models_encoders/df_V2.sav', 'rb')), read_xes(PATH_COMPUTE_CLUSTER))
    t2 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t2)))
    print('execution time (sec):', t2 - t1)
