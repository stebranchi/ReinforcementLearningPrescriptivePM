import time
import pickle
import copy
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from pm4py import read_xes, write_xes
from pm4py.objects.log.obj import Event, EventLog, Trace
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from pm4py.algo.filtering.log.attributes import attributes_filter

PATH = "../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_and_durations.xes"
PATH_COMPUTE_CLUSTER = "../../cluster_data/output_logs/BPI2012_log_eng_all_prefixes.xes"
PATH_CLUSTERED = '../../cluster_data/output_logs/BPI2012_log_eng_clusters100.xes'


def cluster(df, log):
	print("Number of items: " + str(len(df)))

	#filename = '../../cluster_data/models_encoders/df_V2.sav'
	#pickle.dump(df, open(filename, 'wb'), protocol=4)
	#print("saved df")

	#affprop = AffinityPropagation(affinity="euclidean", damping=0.9, max_iter=1000, verbose=True)
	np.nan_to_num(df)
	"""affprop.fit(df)
	#with open("../data/clusters_output/one_hot_resource.txt", "w") as out:
		#out.write(str(one_hot2))
	print("Number of Clusters: " + str(len(np.unique(affprop.labels_))))
	try:
		with open("../data/clusters_output/prova_freq_only_amount_minmax.txt", "w") as output:
			for cluster_id in np.unique(affprop.labels_):
				exemplar = df.transpose()[affprop.cluster_centers_indices_[cluster_id]]
				clusters = [df.transpose()[x] for x in np.nonzero(affprop.labels_==cluster_id)[0]]
				cluster_str = str(clusters)
				output.write(" - *%s:* %s" % (exemplar, cluster_str))
	except Exception as e:
		print("ERROR CLUSTERS:" + str(e))

	filename = '../cluster_data/models_encoders/affinityPropV2.sav'
	pickle.dump(affprop, open(filename, 'wb'), protocol=4)"""

	kmeans = KMeans(n_clusters=100, random_state=0).fit(df)
	trace_clusters_results = dict()
	for i, row in enumerate(df.transpose()):
		trace_clusters_results[row] = kmeans.labels_[i]

	for trace in log:
		trace.attributes["cluster:trace"] = trace_clusters_results[trace.attributes["prefix:name"]]

	write_xes(log, PATH_CLUSTERED)
	filename = '../../cluster_data/models_encoders/kmeansV2_100.sav'
	pickle.dump(kmeans, open(filename, 'wb'), protocol=4)


def computeClusterForTrace(path_affprop, path_kmean, path_df, path_mms, path_onehot, log):
#	with open(path_affprop, 'rb') as f:
#		affprop = pickle.load(f)
#	with open(path_kmean, 'rb') as f:
#		kmean = pickle.load(f)
	with open(path_df, 'rb') as f:
		df = pickle.load(f)
	with open(path_mms, 'rb') as f:
		mms = pickle.load(f)
	with open(path_onehot, 'rb') as f:
		enc = pickle.load(f)

	max_length, events_set, log = prepareLog(log)

	elog = EventLog()
	for trace in log:
		tmp = Trace()
		events_dict = dict()
		amount = 0
		for e in events_set:
			events_dict[e] = 0
		for event in trace:
			tmp.append(event)
			if "AMOUNT_REQ" in trace.attributes.keys():
				amount = int(trace.attributes["AMOUNT_REQ"])
			to_add = list()
			for e, v in events_dict.items():
				to_add.append(v)
			to_add.append(amount)
			if "org:resource" in event.keys():
				to_add.append(str(event["org:resource"]))
			else:
				to_add.append(10000)
			if event["concept:name"] in events_dict.keys():
				events_dict[event["concept:name"]] += 1

			dfline = pd.DataFrame(tuple(to_add)).T
			dfline[23] = mms.transform(dfline[[23]])
			onehot = pd.DataFrame(enc.transform(dfline[[24]]).toarray())
			dfline = dfline.drop(24, axis=1)
			dfline = dfline.astype(float)
			dfline = pd.concat([dfline, onehot], axis=1, ignore_index=True)
			for x in df.transpose():
				if df.transpose()[x].equals(dfline.transpose()):
					#tmp.attributes["affpropCluster"] = affprop.labels_[x]
					#tmp.attributes["kmeanCluster"] = kmean.labels_[x]
					print("correct")
					break
			"""aff_res = affprop.predict(df)
			kmean_res = kmean.predict(df)
			tmp.attributes["affpropCluster"] = aff_res[0]
			tmp.attributes["kmeanCluster"] = kmean_res[0]"""
			elog.append(copy.deepcopy(tmp))

	write_xes(elog, "../../cluster_data/output_logs/BPI2012_testing_20_clusters_all_prefixes.xes")


def filterClusteredLog(log):
	output_log = EventLog()
	for name, value in log.attributes.items():
		output_log.attributes[name] = value
	for name, value in log.extensions.items():
		output_log.extensions[name] = value
	for name, value in log.classifiers.items():
		output_log.classifiers[name] = value
	complete_trace = None
	current_trace_id = log[0].attributes["concept:name"]
	prefixes_cluster = dict()
	for trace in log:
		if trace.attributes["concept:name"] == "198454":
			print("Bella")
		if trace[-1]["concept:name"] == "END":
			complete_trace = trace
		if trace.attributes["concept:name"] != current_trace_id:
			try:
				for prefix_length, cluster_value in prefixes_cluster.items():
					complete_trace[prefix_length-1]["cluster:prefix"] = cluster_value
				output_log.append(complete_trace)
			except:
				print(current_trace_id)
				print(trace.attributes["concept:name"] + '\n\n')
			current_trace_id = trace.attributes["concept:name"]
			prefixes_cluster = dict()
			complete_trace = None
		prefixes_cluster[len(trace)] = trace.attributes["cluster:trace"]

	if complete_trace:
		for prefix_length, cluster_value in prefixes_cluster.items():
			complete_trace[prefix_length-1]["cluster:prefix"] = cluster_value
		output_log.append(complete_trace)

	path = PATH_COMPUTE_CLUSTER.replace("_all_prefixes.xes", "_clusters_squashed.xes")

	write_xes(output_log, path)

def prepareLog(log):
	max_length = 0
	events_set = set()
	for trace in log:
		for e in trace:
			events_set.add(e["concept:name"])
		start = Event()
		start["concept:name"] = "START"
		start["time:timestamp"] = trace[0]["time:timestamp"]
		trace.insert(0, start)
		end = Event()
		end["concept:name"] = "END"
		end["time:timestamp"] = trace[-1]["time:timestamp"]
		trace.append(end)
		if len(trace) > max_length:
			max_length = len(trace)

	return max_length, events_set, log


def createDataFrequency(log):
	# Initializing a new eventlog to save new traces_name for cluster identification
	elog = EventLog()
	# Computing the set of the different events for the columns of the dataframe
	events_set = set()
	for trace in log:
		for e in trace:
			events_set.add(e["concept:name"])
	# Usually the events_set is computed in the next function along with the max_length of trace that is not needed
	# in frequency encoding and the START and END event are already in the log by the precomputation, so we don't need
	# this function for this case
	#max_length, events_set, log = prepareLog(log)
	# Initializing the encoded Dataframe
	data = pd.DataFrame(columns=[x for x in range(len(events_set)+3)])
	i = 0
	for trace in log:
		# Initializing a copy of the trace in order to add each prefix to the dataframe
		tmp = Trace()
		tmp.attributes["concept:name"] = trace.attributes["concept:name"]
		events_dict = dict()
		amount = 0
		# Initializing a dictionary to store the number of occurrences for each event
		for e in events_set:
			events_dict[e] = 0
		for event in trace:
			tmp.append(event)
			# Updating the dictionary
			if event["concept:name"] in events_dict.keys():
				events_dict[event["concept:name"]] += 1
			# Changing trace name to easy find the cluster of the trace
			trace_name = 'trace_%i' % i
			tmp.attributes["prefix:name"] = trace_name
			if "AMOUNT_REQ" in trace.attributes.keys():
				amount = int(trace.attributes["AMOUNT_REQ"])
			# Initializing a new row of the dataset
			to_add = list()
			for e, v in events_dict.items():
				to_add.append(v)
			to_add.append(event["kpi:reward"])
			to_add.append(amount)
			if "org:resource" in event.keys() and event["org:resource"] != 'none':
				to_add.append(str(event["org:resource"]))
			else:
				to_add.append(10000)
			for value in to_add:
				if value is None or value == 'none':
					value = 0
			data.loc[trace_name] = [float(x) for x in to_add]
			elog.append(copy.deepcopy(tmp))
			i+=1
	write_xes(elog, PATH_COMPUTE_CLUSTER)
	# Initializing a MinMaxScaler for each continuous column: reward and amount
	mms_amount = MinMaxScaler()
	mms_reward = MinMaxScaler()
	# Transforming the two continuous columns
	data[len(data.transpose())-2] = mms_amount.fit_transform(data[[len(data.transpose())-2]])
	data[len(data.transpose())-3] = mms_reward.fit_transform(data[[len(data.transpose())-3]])
	# Initializing OneHotEncoder for each categorical column: resource
	enc = OneHotEncoder()
	# Computing dummies for the categorical column
	onehot = pd.DataFrame(enc.fit_transform(data[[len(data.transpose())-1]]).toarray())
	# Drop the column and append the new columns computed by the OneHotEncoder
	data = data.drop(len(data.transpose())-1, axis=1)
	data = data.astype(float)
	onehot.index = data.index
	data = pd.concat([data, onehot], axis=1, ignore_index=True)
	"""
	filename = '../../cluster_data/models_encoders/minMaxScalerAmountV2.sav'
	pickle.dump(mms_amount, open(filename, 'wb'), protocol=4)

	filename = '../../cluster_data/models_encoders/minMaxScalerRewV2.sav'
	pickle.dump(mms_reward, open(filename, 'wb'), protocol=4)

	filename = '../../cluster_data/models_encoders/oneHotEncoderV2.sav'
	pickle.dump(enc, open(filename, 'wb'), protocol=4)"""
	return data


def debugDfValues(path_df):
	with open(path_df, 'rb') as f:
		df = pickle.load(f)

	print(df)

def debugLog(log):
	trace_set = set()
	complete_trace_counter = 0
	for trace in log:
		trace_set.add(trace.attributes["concept:name"])
		if trace[-1]["concept:name"] == "END":
			complete_trace_counter += 1


	print("number of traces: " + str(len(trace_set)))
	print("number of complete traces: " + str(complete_trace_counter))

if __name__ == "__main__":
	t1 = time.time()
	#log = read_xes(PATH)
	#data = createDataFrequency(log)
	#cluster(data, log)
	cluster(pickle.load(open('../../cluster_data/models_encoders/df_V2.sav', 'rb')), read_xes(PATH_COMPUTE_CLUSTER))
	#computeClusterForTrace("../cluster_data/models_encoders/affinityPropNoLastEvent.sav", "../cluster_data/models_encoders/kmeansNoLastEvent.sav", "../cluster_data/models_encoders/minMaxScaler.sav", "../cluster_data/models_encoders/oneHotEncoder.sav", log)
	#computeClusterForTrace("../cluster_data/models_encoders/affinityPropNoLastEvent.sav", "../cluster_data/models_encoders/kmeansNoLastEvent.sav", "../cluster_data/models_encoders/df.sav", "../cluster_data/models_encoders/minMaxScaler.sav", "../cluster_data/models_encoders/oneHotEncoder.sav", log)
	#debugDfValues("../cluster_data/models_encoders/df.sav")
	filterClusteredLog(read_xes(PATH_CLUSTERED))
	#debugLog(read_xes(PATH_CLUSTERED))
	t2 = time.time()
	print(t2-t1)

