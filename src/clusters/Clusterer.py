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

PATH = "../old_final_evaluation/BPI/Log/BPI_2012_log_eng_testing_20.xes"

def cluster(dataset):
	print("Number of items: " + str(len(dataset)))
	df = pd.DataFrame(dataset)
	mms = MinMaxScaler()
	#df = df.rename(columns={0 : "id", 1 : "last_activity", 2 : "n_calls_after_offer", 3 : "n_calls_missing_doc", 4 : "number_of_offers", 5 : "number_of_sent_back", 6 : "W_Fix_incoplete_submission", 7 : "resource"})
	#for i in range(1, 25):
	#	df[i] = mms.fit_transform(df[[i]])
	df[23] = mms.fit_transform(df[[23]])
	enc = OneHotEncoder()
	onehot = pd.DataFrame(enc.fit_transform(df[[24]]).toarray())
	df = df.drop(24, axis=1)
	df = df.astype(float)
	df = pd.concat([df, onehot], axis=1, ignore_index=True)
	#lev_similarity = -1*np.array([[distance(d1, d2) for d1 in dataset] for d2 in dataset])

	affprop = AffinityPropagation(affinity="euclidean", damping=0.8, max_iter=1000, verbose=True)
	affprop.fit(df)
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
	except:
		print("No Clusters to print")

	filename = '../cluster_data/models_encoder/affinityPropNoLastEvent.sav'
	pickle.dump(affprop, open(filename, 'wb'), protocol=4)

	kmeans = KMeans(n_clusters=len(np.unique(affprop.labels_)), random_state=0).fit(df)

	filename = '../cluster_data/models_encoders/kmeansNoLastEvent.sav'
	pickle.dump(kmeans, open(filename, 'wb'), protocol=4)

	filename = '../cluster_data/models_encoders/df.sav'
	pickle.dump(df, open(filename, 'wb'), protocol=4)

	filename = '../cluster_data/models_encoders/minMaxScaler.sav'
	pickle.dump(mms, open(filename, 'wb'), protocol=4)

	filename = '../cluster_data/models_encoders/oneHotEncoder.sav'
	pickle.dump(enc, open(filename, 'wb'), protocol=4)


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


def createData(log):
	data = list()
	max_length = 0
	for trace in log[:500]:
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
	c = 0
	for trace in log[:500]:
		amount = 0
		t = list()
		for event in trace:
			if "AMOUNT_REQ" in trace.attributes.keys():
				amount = trace.attributes["AMOUNT_REQ"]
			to_add = list()
			to_add.append(c)
			to_add.append(event["concept:name"])
			to_add.extend(t)
			to_add.extend(["X"]*(max_length-len(to_add)))
			to_add.append(amount)
			if "org:resource" in event.keys():
				to_add.append(str(event["org:resource"]))
			else:
				to_add.append('R')
			t.append(event["concept:name"])
			data.append(to_add)
			c += 1
	return data

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

def createDataTuple(trace, events_set):
	events_dict = dict()
	amount = 0
	for e in events_set:
		events_dict[e] = 0
	for event in trace:
		if "AMOUNT_REQ" in trace.attributes.keys():
			amount = int(trace.attributes["AMOUNT_REQ"])
		to_add = list()
		to_add.append(event["concept:name"])
		for e, v in events_dict.items():
			to_add.append(v)
		to_add.append(amount)
		if "org:resource" in event.keys():
			to_add.append(str(event["org:resource"]))
		else:
			to_add.append(10000)
		if event["concept:name"] in events_dict.keys():
			events_dict[event["concept:name"]] += 1

def createDataFrequency(log):
	data = set()
	max_length, events_set, log = prepareLog(log)

	for trace in log:
		events_dict = dict()
		amount = 0
		for e in events_set:
			events_dict[e] = 0
		for event in trace:
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
			data.add(tuple(to_add))
	return list(data)


def distance(d1,d2):
	for x,y in zip(d1, d2):
		score = 0
		if isinstance(x, str) and isinstance(y, str):
			if x != y:
				score += 1
		elif (isinstance(x, float) or isinstance(x, int)) and (isinstance(y, float) or isinstance(y, int)):
			score += int((int(y)-int(x))/100)
		else:
			print("Type Error")

	return -1 * score


if __name__ == "__main__":
	t1 = time.time()
	log = pm4py.read_xes(PATH)
	#data = createDataFrequency(log)
	#cluster(data)
	#computeClusterForTrace("../cluster_data/models_encoders/affinityPropNoLastEvent.sav", "../cluster_data/models_encoders/kmeansNoLastEvent.sav", "../cluster_data/models_encoders/minMaxScaler.sav", "../cluster_data/models_encoders/oneHotEncoder.sav", log)
	computeClusterForTrace("../cluster_data/models_encoders/affinityPropNoLastEvent.sav", "../cluster_data/models_encoders/kmeansNoLastEvent.sav", "../cluster_data/models_encoders/df.sav", "../cluster_data/models_encoders/minMaxScaler.sav", "../cluster_data/models_encoders/oneHotEncoder.sav", log)
	t2 = time.time()
	print(t2-t1)

