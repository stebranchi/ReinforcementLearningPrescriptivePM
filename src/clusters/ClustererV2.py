import pm4py
import time
import pickle
import copy
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from pm4py.objects.log.obj import Event, EventLog, Trace
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

PATH = "../data/BPI2013/BPI_2012_log_eng_rewards_and_durations.xes"

def cluster(df):
	print("Number of items: " + str(len(df)))

	filename = '../cluster_data/models_encoders/df_base.sav'
	pickle.dump(df, open(filename, 'wb'), protocol=4)
	print("saved df")

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

	filename = '../cluster_data/models_encoders/affinityPropBase.sav'
	pickle.dump(affprop, open(filename, 'wb'), protocol=4)"""

	kmeans = KMeans(n_clusters=300, random_state=0).fit(df)

	filename = '../cluster_data/models_encoders/kmeansBase.sav'
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

	pm4py.write_xes(elog, "../cluster_data/output_logs/BPI2012_testing_20_clusters_all_prefixes.xes")


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
			tmp.attributes["concept:name"] = trace_name
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
	# pm4py.write_xes(elog, "../cluster_data/output_logs/BPI2012_log_eng_all_prefixes.xes")
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
	filename = '../cluster_data/models_encoders/minMaxScalerAmountBase.sav'
	pickle.dump(mms_amount, open(filename, 'wb'), protocol=4)

	filename = '../cluster_data/models_encoders/minMaxScalerRewBase.sav'
	pickle.dump(mms_reward, open(filename, 'wb'), protocol=4)

	filename = '../cluster_data/models_encoders/oneHotEncoderBase.sav'
	pickle.dump(enc, open(filename, 'wb'), protocol=4)"""
	return data


def debugDfValues(path_df):
	with open(path_df, 'rb') as f:
		df = pickle.load(f)

	print(df)


if __name__ == "__main__":
	t1 = time.time()
	log = pm4py.read_xes(PATH)
	data = createDataFrequency(log)
	#cluster(data)
	#computeClusterForTrace("../cluster_data/models_encoders/affinityPropNoLastEvent.sav", "../cluster_data/models_encoders/kmeansNoLastEvent.sav", "../cluster_data/models_encoders/minMaxScaler.sav", "../cluster_data/models_encoders/oneHotEncoder.sav", log)
	#computeClusterForTrace("../cluster_data/models_encoders/affinityPropNoLastEvent.sav", "../cluster_data/models_encoders/kmeansNoLastEvent.sav", "../cluster_data/models_encoders/df.sav", "../cluster_data/models_encoders/minMaxScaler.sav", "../cluster_data/models_encoders/oneHotEncoder.sav", log)
	#debugDfValues("../cluster_data/models_encoders/df.sav")
	t2 = time.time()
	print(t2-t1)

