import os
import numpy as np
import itertools
import pm4py
import distance

from datetime import datetime
from sklearn.cluster import AffinityPropagation
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.importer.xes import importer as xes_importer


def smithWaterman(a, b, match_score=3, gap_cost=2):
	H = np.zeros((len(a) + 1, len(b) + 1), np.int)

	for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
		#match = H[i - 1, j - 1] + (10 - distance.levenshtein(a[i - 1], b[j - 1]))
		match_score = -3
		if a[i - 1] == b[j - 1]:
			match_score = 3
		elif a[i - 1].split('-')[0].strip() == b[j - 1].split('-')[0].strip():
			match_score = 2

		match = H[i - 1, j - 1] + match_score
		delete = H[i - 1, j] - gap_cost
		insert = H[i, j - 1] - gap_cost
		H[i, j] = max(match, delete, insert, 0)
		#H[i, j] = match

	H_flip = np.flip(np.flip(H, 0), 1)
	i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
	i, j = np.subtract(H.shape, (i_ + 1, j_ + 1))
	return H[i, j]


def event_attribute_to_state(event, attribute):
	event = event.upper()
	if event in ('START', 'END'):
		state = '<' + event + '>'
	else:
		attribute_clean = attribute.replace('-','').replace('0','').upper()
		state = '<' + event + ' - ' + attribute_clean + '>'
	return state


def calculatedistance(w1, w2):
	dist = 0
	for x, y in zip(w1,w2):
		dist += distance.levenshtein(x, y)

	return dist


def createclusters(test_log, prefix, test_trace):
	ids_list = list()
	clusters_dict = dict()
	words = set()
	test_trace = [x["concept:name"] for x in test_trace[1:prefix+1]]
	#test_trace = [event_attribute_to_state(x["concept:name"], x["Resource"]) for x in log[0][1:prefix]]
	# STESSI PREFISSI
	"""for trace in test_log:
		t_str = list()
		for event in trace:
			if event["concept:name"] not in ("START", "END"):
				#t_str.append(event_attribute_to_state(event["concept:name"], event["Resource"]))
				t_str.append(event["concept:name"])
		words.append(t_str[:prefix])"""
		#words.append(''.join(t_str[:prefix + 1]))

	#CON TUTTI I PREFISSI
	for trace in test_log:
		for i in range(1, len(trace)):
			words.add(tuple([x["concept:name"] for x in trace[:i]]))

	words = np.asarray(list(words)) #So that indexing with a list will work
	#lev_similarity = -1*np.array([[calculatedistance(w1, w2) for w1 in words] for w2 in words])
	lev_similarity = np.array([[smithWaterman(w1, w2) for w1 in words] for w2 in words])
	#lev_similarity = -1*np.array([[distance.levenshtein(w1, w2) for w1 in words] for w2 in words])

	affprop = AffinityPropagation(affinity="euclidean", damping=0.5)
	affprop.fit(lev_similarity)

	for cluster_id in affprop.labels_:
		exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
		cluster = words[np.nonzero(affprop.labels_ == cluster_id)]
		clusters_dict[str(exemplar)] = cluster

	print(str(clusters_dict))
	test_similarity = np.array([smithWaterman(tuple(test_trace), w1) for w1 in words])
	test_cluster = affprop.predict(test_similarity.reshape(1, -1))[0]
	for trace in test_log:
		#tmp_trace = [event_attribute_to_state(x["concept:name"], x["Resource"]) for x in trace[1:prefix]]
		tmp_trace = [x["concept:name"] for x in trace[1:prefix+1]]
		if str(test_trace) in clusters_dict.keys():
			if tmp_trace in clusters_dict[str(test_trace)]:
				ids_list.append(trace.attributes["concept:name"])
		elif tmp_trace == test_trace:
			ids_list.append(trace.attributes["concept:name"])

	return ids_list



#fare una funzione custom su lista di elementi, scansionare il log per le tracce che soddisfano e poi ritornare gli id traccia
#aggiungere flag per risorsa oppure no

def main():
	# import log
	log_path = os.path.join("data", "logs", "simple_model+changes_only_completed.xes")
	log = xes_importer.apply(log_path)
	test_log = attributes_filter.apply_events(log, ["complete"],
											  parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition", attributes_filter.Parameters.POSITIVE: True})

	prefix = 3
	#get clusters from prefixes
	clusters = createclusters(test_log, prefix, test_log[1])
	print(clusters)


if __name__ == "__main__":
	main()

