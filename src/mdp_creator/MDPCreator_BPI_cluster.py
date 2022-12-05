import os
import math
import numpy

import pandas as pd
import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import Event
from sklearn.preprocessing import MinMaxScaler
import copy
import datetime

def main(path, tracefilter_log_pos):
	OUTPUT_PATH = path.replace("logs","mdps").replace('.xes', '_not_scaled.csv')
	cols_headers = ["s","a","s'","p_r","reward","number_occurrences"]
	#tracefilter_log_pos = pm4py.read_xes(path)
	reward_dict = dict()

	#WORKING VERSION
	dfg_dict = dict()
	for trace in tracefilter_log_pos:
		event_name = getEventResourceName_cluster(trace[0], trace)
		for event in trace:
			if event['concept:name'] != 'START':
				new_event_name = getEventResourceName_cluster(event, trace)
				key = (event_name, new_event_name)
				reward_dict[key] = {'count': 0, 'sum': 0.0, 'reward': 0}
				event_name = new_event_name
				if key in dfg_dict.keys():
					dfg_dict[key] += 1
				else:
					dfg_dict[key] = 1

	for trace in tracefilter_log_pos:
		start = trace[0]['time:timestamp']
		event_name = getEventResourceName_cluster(trace[0], trace)
		for event in trace:
			if event['concept:name'] not in ['START']:
				new_event_name = getEventResourceName_cluster(event, trace)
				if "cluster:reward" in event.keys():
					reward_dict[(event_name, new_event_name)]["sum"] += float(event["cluster:reward"])
					reward_dict[(event_name, new_event_name)]["count"] += 1
					event_name = new_event_name
				else:
					reward_dict[(event_name, new_event_name)]["sum"] += 0
					reward_dict[(event_name, new_event_name)]["count"] += 1
					event_name = new_event_name

	max_occurrences = 0
	for key, item in reward_dict.items():
		if item["count"] > max_occurrences:
			max_occurrences = item["count"]
		item['reward'] = item['sum'] / item['count'] if item['count'] > 0 else 0

	csv = pd.DataFrame(columns=cols_headers)
	starts = pd.DataFrame(columns=cols_headers)
	csv_dict = dict()
	max_count = dict()

	for (s1, s2) in dfg_dict.keys():
		if "AZIONE" in s1:
			s1_csv = "<" + s1.split("AZIONE")[-1]
		else:
			s1_csv = s1
		csv_dict[s1_csv] = {}

	"""new_dfg_dict = copy.deepcopy(dfg_dict)
	for key, c in dfg_dict.items():
		if c < 50:
			del(new_dfg_dict[key])"""

	for (s1, s2), c in dfg_dict.items():
		reward = 0
		if "AZIONE" in s1:
			s1_csv = "<" + s1.split("AZIONE")[-1]
		else:
			s1_csv = s1
		if "AZIONE" in s2:
			parts = s2.split("AZIONE")
			s2_csv = "<" + parts[-1]
			a = parts[0].replace("<", "").replace(">","")
		else:
			s2_csv = s2
			a = s2.split(" | ")[0].replace("<", "").replace(">","").rstrip()
		if 'END' in s2:
			if (a, s2_csv) in csv_dict[s1_csv].keys():
				csv_dict[s1_csv][(a, s2_csv)]["c"] = csv_dict[s1_csv][(a, s2_csv)]["c"] + c
			else:
				rew = reward_dict[(s1, s2)]['reward']
				csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": rew}
				#csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": -2*math.exp(-c/3)/(1+math.exp(-c/3))*max_occurrences + rew}
		else:
			csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": 0}
			#csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": -2*math.exp(-c/3)/(1+math.exp(-c/3))*max_occurrences}

	for key, item in csv_dict.items():
		for (k1, k2), t in item.items():
			if (key, k1) in max_count.keys():
				max_count[(key, k1)] += t["c"]
			else:
				max_count[(key, k1)] = t["c"]

	max_c = max([v for k,v in max_count.items()])
	with open(OUTPUT_PATH.replace("output_mdps", "logs_stats"), 'w') as out_stats:
		values = numpy.array([v for k,v in max_count.items()]).astype(float)
		out_stats.write("Min: " + str(min(values)) + "\n")
		out_stats.write("Max: " + str(max(values)) + "\n")
		out_stats.write("Average: " + str(numpy.mean(values)) + "\n")
		out_stats.write("Median: " + str(numpy.median(values)) + "\n")
		out_stats.write("Q1: " + str(numpy.quantile(values, 0.25)) + "\n")
		out_stats.write("Q3: " + str(numpy.quantile(values, 0.75)) + "\n")
	"""csv_to_rem = dict()
	for (s1, s2), v in max_count.items():
		if "START" not in s1 and "END" not in s2:
			out_sum = sum([v for (k1, k2), v in max_count.items() if k1 == s1])
			in_sum =  sum([v for (k1, k2), v in max_count.items() if k2 == s2])
			if v < 20:
				csv_to_rem[(s1, s2)] = (v, in_sum, out_sum)

	for (k1, k2) in csv_to_rem.keys():
		del(max_count[(k1, k2)])"""

	row = ["<>","START","<START>",1.0,0.0,max_c]
	starts = starts.append({k:v for (k,v) in zip(cols_headers,row)}, ignore_index=True)
	for key, item in csv_dict.items():
		for (k1, k2), t in item.items():
			if (key, k1) in max_count.keys():
				row = [key, k1, k2, t["c"]/max_count[(key, k1)], t["rew"], t["c"]]
				if key == '<START>':
					starts = starts.append({k:v for (k,v) in zip(cols_headers,row)}, ignore_index=True)
				else:
					csv = csv.append({k:v for (k,v) in zip(cols_headers,row)}, ignore_index=True)

	csv = csv.sort_values('s')
	starts = starts.sort_values('s')
	full = starts.append(csv)
	mmrew = MinMaxScaler().fit(full[["reward"]])
	mmocc = MinMaxScaler().fit(full[["number_occurrences"]])
	#out = open(OUTPUT_PATH, 'w')
	#out.write("s,a,s',p_r,reward,number_occurrences\n<>,START,<START>,1.0,0.0," + str(max_c) + "\n")
	full["reward"] = full["reward"].map(lambda x: 0 if x == 0 else mmrew.transform([[x]])[0][0])
	# full["number_occurrences"] = full["number_occurrences"].map(lambda x: 0 if x == 0 else mmocc.transform([[x]])[0][0])
	full.to_csv(OUTPUT_PATH, index=False)
	"""for row in starts.transpose():
		out.write(','.join([str(x) for x in starts.transpose()[row]]) + '\n')
	for row in csv.transpose():
		out.write(','.join([str(x) for x in csv.transpose()[row]]) + '\n')"""
	#out.close()

def getEventResourceName_cluster(event, trace):
	if event["concept:name"] in 'START':
		return "<" + event["concept:name"] + ">"
	else:
		try:
			return "<" + event["concept:name"] + " | " + event["cluster:prefix"] + ">"
		except:
			print(event)
			print(trace.attributes["concept:name"] + '\n\n')


def getEventResourceName_no_wfix(event):
	if event["concept:name"] == 'START' or event["concept:name"] == 'END':
		return "<" + event["concept:name"] + ">"
	if "amount" in event.keys():
		return "<" + event["concept:name"] + " - " + str(event["amount"]) + " - " + str(event["n_calls_after_offer"]) + " - " + str(event["n_calls_missing_doc"]) + " - " + str(event["number_of_offers"]) + " - " + str(event["number_of_sent_back"]) + ">"
	else:
		return "<" + event["concept:name"] + ">"


def getEventResourceName(event):
	if event["concept:name"] == 'START' or event["concept:name"] == 'END':
		return "<" + event["concept:name"] + ">"
	if "amount" in event.keys():
		return "<" + event["concept:name"] + " - " + str(event["amount"]) + " - " + str(event["n_calls_after_offer"]) + " - " + str(event["n_calls_missing_doc"]) + " - " + str(event["number_of_offers"]) + " - " + str(event["number_of_sent_back"]) + " - " + str(event["W_Fix_incoplete_submission"]) + ">"
	else:
		return "<" + event["concept:name"] + ">"

def preprocessBPI(path):
	output_path = path.replace('.xes', '_preprocessed.xes')
	mid_output_path = path.replace('.xes', '_mid_preprocessed.xes')
	log = pm4py.read_xes(path)
	begin_event = Event()
	begin_event["concept:name"] = ""

	#filtered_log = variants_filter.filter_variants_by_coverage_percentage(log, min_coverage_percentage=0.0001)

	for trace in log:
		last_event = Event()
		last_event["concept:name"] = ""
		for event in trace:
			if event["concept:name"] in ["O_ACCEPTED", "A_CANCELLED", "O_SENT_BACK"]:
				idx = [x for x, e in enumerate(trace) if e == event][0] - 1
				event["concept:name"] = trace[idx]["concept:name"] + "AZIONE" + event["concept:name"]
				trace[idx]["concept:name"] = "TO_REMOVE"
			if last_event["concept:name"] == event["concept:name"] and event["concept:name"] is not "TO_REMOVE":
				if "duration" in last_event.keys() and "duration" in event.keys():
					event["duration"] = event["duration"] + last_event["duration"]
				idx = [x for x, e in enumerate(trace) if e == last_event][0]
				trace[idx]["concept:name"] = "TO_REMOVE"
			last_event = event


	tracefilter_log_pos = attributes_filter.apply_events(log, ["TO_REMOVE"],
														 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	xes_exporter.apply(tracefilter_log_pos, output_path)

	"""for trace in tracefilter_log_pos:
		for event in trace:
			if "AZIONE" in event["concept:name"]:
				event["concept:name"] = event["concept:name"].split("AZIONE")[-1]
			if "duration" not in event.keys():
				event["duration"] = 0

	xes_exporter.apply(tracefilter_log_pos, mid_output_path)"""
	print("end preprocessing")
	return output_path, tracefilter_log_pos

def checkMDPS(path1, path2):
	with open(path1, 'r') as csv1:
		states_csv1 = [line.split(",")[0] for line in csv1]
	with open(path2, 'r') as csv2:
		states_csv2 = [line.split(",")[0] for line in csv2]

	states_only_test = set(states_csv1) - set(states_csv2)

	print(states_only_test)


if __name__ == "__main__":
	path = "../../cluster_data/node2vec/output_logs/BPI_2012_log_eng_ordered_training_80_edgeavgmax8_35.xes"
	output_path, output_log = preprocessBPI(path)
	#output_path = "../../cluster_data/output_logs/BPI2012_new_arch_positional_cumulative_squashed_training_80_preprocessed.xes"
	if "training" in path:
		main(output_path, output_log)
	#checkMDPS("../../cluster_data/output_mdps/BPI2012_new_arch_positional_cumulative_squashed_testing_20_preprocessed_scaled.csv",
	#		  "../../cluster_data/output_mdps/BPI2012_new_arch_positional_cumulative_squashed_training_80_preprocessed_scaled.csv")
