import time
import pickle
import numpy as np
import pandas as pd
from pm4py import read_xes, write_xes
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from statistics import mean, median, stdev

N_CLUSTERS = 95
TRAINING_PATH = "../../data/logs/BPI_2012/BPI_2012_log_eng_ordered_nomicro_rewards_cumulative_durations_training_80.xes"
TESTING_PATH = "../../data/logs/BPI_2012/BPI_2012_log_eng_ordered_nomicro_rewards_cumulative_durations_testing_20.xes"
TRAINING_ENCODING_FILENAME = "BPI_2012_log_eng_ordered_training_80_edgeavgmax8.csv"
TESTING_ENCODING_FILENAME = "BPI_2012_log_eng_ordered_testing_20_edgeavgmax8.csv"

TRAINING_ENCODING = "../../cluster_data/node2vec/node2vec/" + TRAINING_ENCODING_FILENAME
TESTING_ENCODING = "../../cluster_data/node2vec/node2vec/" + TESTING_ENCODING_FILENAME
PATH_CLUSTERED_TRAINING = "../../cluster_data/node2vec/output_logs/" + TRAINING_ENCODING_FILENAME.replace(".csv", "_" + str(N_CLUSTERS) + ".xes")
PATH_CLUSTERED_TESTING = "../../cluster_data/node2vec/output_logs/" + TESTING_ENCODING_FILENAME.replace(".csv", "_" + str(N_CLUSTERS) + ".xes")


def prepare_log(df, log, training=True, mms=None):
    if training:
        trace_to_index = {str(t.attributes["concept:name"]): i for i, t in enumerate(log)}
        events_set = set([e["concept:name"] for t in log for e in t])
        last_act_to_cluster = {x: set() for x in events_set}
        mms = MinMaxScaler(feature_range=(0, 1))
        df["label"] = mms.fit_transform(df[["label"]])
        return df, last_act_to_cluster, mms, trace_to_index
    else:
        trace_to_index = {str(t.attributes["concept:name"]): i for i, t in enumerate(log)}
        df["label"] = mms.transform(df[["label"]])
        return df, trace_to_index


def stats(list):
   stats = {'max': max(list), 'avg': mean(list), 'median': median(list), 'stdev': stdev(list),
                         'Q3': np.percentile(list, 75), 'D9': np.percentile(list, 90)}
   return stats


def cluster(df, log, mms, last_a_to_cluster, trace_to_index):
    print("Number of items: " + str(len(df)))
    np.nan_to_num(df)
    columns = list(df.columns)
    columns.remove("case")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(df[columns])
    print("Finished Clustering, now starting exporting results")
    df.apply(iterTrainingDf, args=(log, last_a_to_cluster, mms, kmeans, trace_to_index), axis=1)
    write_xes(log, PATH_CLUSTERED_TRAINING)
    print("Finished writing training log")
    #filename = '../../cluster_data_old/models_encoders/kmeans_model_new.sav'
    #pickle.dump(kmeans, open(filename, 'wb'), protocol=4)

    return kmeans


def fit_test_data(df, log, mms, kmean, last_a_to_cluster, trace_to_index):
    print("Number of items: " + str(len(df)))
    np.nan_to_num(df)
    columns = list(df.columns)
    columns.remove("case")
    labels = kmean.transform(df[columns])
    print("Finished Clustering, now starting exporting results")
    df.apply(iterTestingDf, args=(log, last_a_to_cluster, mms, kmean, labels, trace_to_index), axis=1)

    write_xes(log, PATH_CLUSTERED_TESTING)
    print("Finished writing testing log")


def iterTrainingDf(row, log, last_a_to_cluster, mms, kmeans, trace_to_index):
    parts = row["case"].split("_")
    event = log[trace_to_index[parts[0]]][int(parts[1])-1]
    label = kmeans.labels_[row.name]
    event["cluster:prefix"] = label
    event["stato"] = event["concept:name"] + " | " + str(label)
    event["cluster:reward"] = mms.inverse_transform([[kmeans.cluster_centers_[label][-1]]])[0][0]
    last_a_to_cluster[event["concept:name"]].add(label)


def iterTestingDf(row, log, last_a_to_cluster, mms, kmean, labels, trace_to_index):
    parts = row["case"].split("_")
    event = log[trace_to_index[parts[0]]][int(parts[1])-1]
    best_cluster_value, best_cluster = min([(labels[row.name][x], x) for x in last_a_to_cluster[event["concept:name"]]],
                                           key=lambda x: x[0])
    event["cluster:prefix"] = best_cluster
    event["stato"] = event["concept:name"] + " | " + str(best_cluster)
    event["cluster:reward"] = mms.inverse_transform([[kmean.cluster_centers_[best_cluster][-1]]])[0][0]


if __name__ == "__main__":
    t1 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t1)))
    training_log = read_xes(TRAINING_PATH)
    train_df = pd.read_csv(TRAINING_ENCODING)
    training_data, last_act_to_cluster, training_mms, trace_to_i = prepare_log(train_df, training_log)
    kmeans = cluster(training_data, training_log, training_mms, last_act_to_cluster, trace_to_i)
    testing_log = read_xes(TESTING_PATH)
    test_df = pd.read_csv(TESTING_ENCODING)
    testing_data, trace_to_i = prepare_log(test_df, testing_log, False, training_mms)
    fit_test_data(testing_data, testing_log, training_mms, kmeans, last_act_to_cluster, trace_to_i)
    # cluster(pickle.load(open('../../cluster_data/models_encoders/df_V2.sav', 'rb')), read_xes(PATH_COMPUTE_CLUSTER))
    t2 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t2)))
    print(t2 - t1)
