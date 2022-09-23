import os
import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import Event
import copy
import datetime

def main(path):
	OUTPUT_PATH = path.replace("logs","mdps").replace('.xes', '.csv')
	tracefilter_log_pos = pm4py.read_xes(path)
	reward_dict = dict()

	#WORKING VERSION
	dfg_dict = dict()
	for trace in tracefilter_log_pos:
		event_name = getEventResourceName_cluster(trace[0], trace)
		for event in trace:
			if event['concept:name'] != 'START':
				new_event_name = getEventResourceName_cluster(event, trace)
				key = (event_name, new_event_name)
				reward_dict[key] = {'count': 0, 'sum': 0, 'reward': 0}
				event_name = new_event_name
				if key in dfg_dict.keys():
					dfg_dict[key] += 1
				else:
					dfg_dict[key] = 1

	for trace in tracefilter_log_pos:
		start = trace[0]['time:timestamp']
		event_name = getEventResourceName_cluster(trace[0], trace)
		for event in trace:
			if event['concept:name'] != 'START' and event['concept:name'] != 'END':
				new_event_name = getEventResourceName_cluster(event, trace)
				if "kpi:reward" in event.keys():
					reward_dict[(event_name, new_event_name)]["sum"] += int(event["kpi:reward"])
					reward_dict[(event_name, new_event_name)]["count"] += 1
					event_name = new_event_name
				else:
					reward_dict[(event_name, new_event_name)]["sum"] += 0
					reward_dict[(event_name, new_event_name)]["count"] += 1
					event_name = new_event_name

	for key, item in reward_dict.items():
		item['reward'] = item['sum'] / item['count'] if item['count'] > 0 else 0

	csv = list()
	starts = list()
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
				csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": 0.0}
		else:
			csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": reward_dict[(s1, s2)]['reward']}

	for key, item in csv_dict.items():
		for (k1, k2), t in item.items():
			if (key, k1) in max_count.keys():
				max_count[(key, k1)] += t["c"]
			else:
				max_count[(key, k1)] = t["c"]

	"""csv_to_rem = dict()
	for (s1, s2), v in max_count.items():
		if "START" not in s1 and "END" not in s2:
			out_sum = sum([v for (k1, k2), v in max_count.items() if k1 == s1])
			in_sum =  sum([v for (k1, k2), v in max_count.items() if k2 == s2])
			if v < 20:
				csv_to_rem[(s1, s2)] = (v, in_sum, out_sum)

	for (k1, k2) in csv_to_rem.keys():
		del(max_count[(k1, k2)])"""

	for key, item in csv_dict.items():
		for (k1, k2), t in item.items():
			if (key, k1) in max_count.keys():
				if key == '<START>':
					starts.append([key, k1, k2, t["c"]/max_count[(key, k1)], t["rew"], t["c"]])
				else:
					csv.append([key, k1, k2, t["c"]/max_count[(key, k1)], t["rew"], t["c"]])

	csv = sorted(csv)
	starts = sorted(starts)
	out = open(OUTPUT_PATH, 'w')
	out.write("s, a, s', p_r, duration, number_occurrences\n<>,START,<START>,1.0,0.0,0\n")
	for row in starts:
		out.write(','.join([str(x) for x in row]) + '\n')
	for row in csv:
		out.write(','.join([str(x) for x in row]) + '\n')
	out.close()

def getEventResourceName_cluster(event, trace):
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

	for trace in tracefilter_log_pos:
		for event in trace:
			if "AZIONE" in event["concept:name"]:
				event["concept:name"] = event["concept:name"].split("AZIONE")[-1]

	xes_exporter.apply(tracefilter_log_pos, mid_output_path)
	print("end preprocessing")
	return output_path

if __name__ == "__main__":
	output_path = preprocessBPI("../../cluster_data/output_logs/BPI2012_log_eng_clusters_squashed_testing_20.xes")
	#output_path = "../../cluster_data/output_logs/BPI2012_log_eng_clusters_squashed_training_80_preprocessed.xes"
	main(output_path)
