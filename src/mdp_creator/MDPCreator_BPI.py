import os
import pm4py
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import Event
import copy
import datetime

def main():
	PATH = os.path.join("..", "final_evaluation", "BPI", "Log", "BPI_2012_log_eng_training_80_preprocessed_number_incr_wfix.xes")
	OUTPUT_PATH = PATH.replace("Log","Mdp").replace('.xes', '.csv')
	tracefilter_log_pos = pm4py.read_xes(PATH)
	#start_activities = pm4py.get_start_activities(log)
	#end_activities = pm4py.get_end_activities(log)
	duration_dict = dict()
	reward_factor = 3
	resources_per_event = dict()
	#tracefilter_log_pos = attributes_filter.apply_events(log, ["COMPLETE", "complete"],
	#													 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition", attributes_filter.Parameters.POSITIVE: True})

	#WORKING VERSION
	dfg_dict = dict()
	for trace in tracefilter_log_pos:
		event_name = getEventResourceName(trace[0])
		for event in trace:
			if event['concept:name'] != 'START':
				new_event_name = getEventResourceName(event)
				key = (event_name, new_event_name)
				duration_dict[key] = {'starts': list(), 'ends': list(), 'count': 0, 'sum': 0, 'duration': 0}
				event_name = new_event_name
				if key in dfg_dict.keys():
					dfg_dict[key] += 1
				else:
					dfg_dict[key] = 1

	for trace in tracefilter_log_pos:
		start = trace[0]['time:timestamp']
		event_name = getEventResourceName(trace[0])
		for event in trace:
			if event['concept:name'] != 'START' and event['concept:name'] != 'END':
				new_event_name = getEventResourceName(event)
				if "duration" in event.keys():
					duration_dict[(event_name, new_event_name)]["sum"] += int(event["duration"])
					duration_dict[(event_name, new_event_name)]["count"] += 1
					start = event['time:timestamp']
					event_name = new_event_name
				else:
					"""duration_dict[(event_name, new_event_name)]['starts'].append(start)
					start = event['time:timestamp']
					duration_dict[(event_name, new_event_name)]['ends'].append(start)
					event_name = new_event_name"""
					duration_dict[(event_name, new_event_name)]["sum"] += 0
					duration_dict[(event_name, new_event_name)]["count"] += 1
					start = event['time:timestamp']
					event_name = new_event_name

	for key, item in duration_dict.items():
		for s, e in zip(item['starts'], item['ends']):
			item['sum'] += (e-s).total_seconds()
			item['count'] += 1
		item['starts'] = item['ends'] = []
		item['duration'] = item['sum'] / item['count'] if item['count'] > 0 else 0

	#parameters = {}
	#dfg = dfg_discovery.apply(tracefilter_log_pos, parameters=None)

	#net, im, fm = dfg_mining.apply(dfg)
	csv = list()
	starts = list()
	csv_dict = dict()
	#dfg_dict = dict(dfg)
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
			a = s2.split(" - ")[0].replace("<", "").replace(">","").rstrip()
		if "O_ACCEPTED" in s2:
			if s1.split(" - ")[1] == "low":
				reward = 650
			elif s1.split(" - ")[1] == "medium":
				reward = 1900
			elif s1.split(" - ")[1] == "high":
				reward = 5900
		if 'END' in s2:
			if (a, s2_csv) in csv_dict[s1_csv].keys():
				csv_dict[s1_csv][(a, s2_csv)]["c"] = csv_dict[s1_csv][(a, s2_csv)]["c"] + c
			else:
				csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": 0.0}
		else:
			#rew = (c/reward_factor)**2/(1+(c/reward_factor)**2) * reward
			#x = (-duration_dict[(s1, s2)]['duration'] * 0.005) + rew
			x = (-duration_dict[(s1, s2)]['duration'] * 0.005) + reward
			csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": x}

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

def preprocessBPI():
	PATH = os.path.join("..", "final_evaluation", "BPI", "Log", "BPI_2012_log_eng_testing_20.xes")
	OUTPUT_PATH = PATH.replace('.xes', '_preprocessed_number_incr_wfix.xes')
	MID_OUTPUT_PATH = PATH.replace('.xes', '_mid_preprocessed_number_incr_wfix.xes')
	attributes_upper_limit = 5
	log = pm4py.read_xes(PATH)
	begin_event = Event()
	begin_event["concept:name"] = ""

	filtered_log = variants_filter.filter_log_variants_percentage(log, percentage=0.9)

	#tracefilter_log_pos = attributes_filter.apply(log, ["W_Assess_fraud"],
	#											  parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})
	tracefilter_log_pos = attributes_filter.apply(filtered_log, ["W_Assess_fraud"],
													  parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	tracefilter_log_pos_2 = attributes_filter.apply_events(tracefilter_log_pos, ["SCHEDULE"],
														 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition", attributes_filter.Parameters.POSITIVE: False})

	tracefilter_log_pos_3 = attributes_filter.apply_events(tracefilter_log_pos_2, ["W_Complete_preaccepted_application", "A_APPROVED", "A_REGISTERED", "A_ACTIVATED", "A_FINALIZED"],
														 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	#, "A_ACCEPTED", "A_FINALIZED", "A_PARTLYSUBMITTED", "A_PREACCEPTED"
	for trace in tracefilter_log_pos_3:
		amount_req = int(trace.attributes["AMOUNT_REQ"])
		if amount_req < 6000:
			amount = "low"
		elif amount_req < 15000:
			amount = "medium"
		else:
			amount = "high"
		start = Event()
		start["concept:name"] = "START"
		start["time:timestamp"] = trace[0]["time:timestamp"]
		trace.insert(0, start)
		end = Event()
		end["concept:name"] = "END"
		end["time:timestamp"] = trace[-1]["time:timestamp"]
		trace.append(end)
		last_event = Event()
		last_event["concept:name"] = ""
		for event in trace:
			event["time:timestamp"] = event["time:timestamp"].isoformat(timespec='milliseconds')
			if event["concept:name"] not in ["START", "END"]:
				if "org:resource" not in event.keys():
					event["org:resource"] = "none"
				event["amount"] = amount
				if event["concept:name"].startswith("W_") and event["lifecycle:transition"] == "START":
					begin_event = event
				if event["concept:name"] == begin_event["concept:name"] and event["lifecycle:transition"] == "COMPLETE":
					duration = datetime.datetime.strptime(event["time:timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z") - datetime.datetime.strptime(begin_event["time:timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z")
					# controllare due giorni diversi invece che durata
					if datetime.datetime.strptime(event["time:timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").day is not datetime.datetime.strptime(begin_event["time:timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").day:
						begin_event["duration"] = 2000
					else:
						begin_event["duration"] = duration.total_seconds()
					begin_event = Event()
					begin_event["concept:name"] = ""
				if event["concept:name"] in ["W_Call_after_offer", "W_Call_missing_information", "W_Assess_application", "W_Fix_incoplete_submission", "W_Complete_preaccepted_appl", "O_CANCELLED", "O_SELECTED"] and event["lifecycle:transition"] == "COMPLETE":
					event["concept:name"] = "TO_REMOVE"
				if event["concept:name"] in ["A_APPROVED", "A_REGISTERED", "A_ACTIVATED"] and event["lifecycle:transition"] == "COMPLETE":
					event["concept:name"] = "O_ACCEPTED"
				if last_event["concept:name"] == event["concept:name"] == "O_ACCEPTED":
					idx = [x for x, e in enumerate(trace) if e == last_event][0]
					trace[idx]["concept:name"] = "TO_REMOVE"
				if last_event["concept:name"] == event["concept:name"] and event["concept:name"] is not "TO_REMOVE":
					if "duration" in last_event.keys() and "duration" in event.keys():
						event["duration"] = event["duration"] + last_event["duration"]
					idx = [x for x, e in enumerate(trace) if e == last_event][0]
					trace[idx]["concept:name"] = "TO_REMOVE"
			last_event = event

	tracefilter_log_pos_4 = attributes_filter.apply_events(tracefilter_log_pos_3, ["TO_REMOVE"],
														 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	for trace in tracefilter_log_pos_4:
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

	for trace in tracefilter_log_pos_4:
		calls_after_no = 0
		calls_missing_no = 0
		offers_no = 0
		sent_back = 0
		w_fix = 0
		for event in trace:
			if "O_SENT" in event["concept:name"]:
				offers_no += 1
			if "W_Call_after_offer" in event["concept:name"]:
				calls_after_no += 1
			if "W_Call_missing_information" in event["concept:name"]:
				calls_missing_no += 1
			if "O_SENT_BACK" in event["concept:name"]:
				sent_back += 1
			if "W_Fix_incoplete_submission" in event["concept:name"]:
				w_fix += 1
			event["n_calls_after_offer"] = min(calls_after_no, attributes_upper_limit)
			event["n_calls_missing_doc"] = min(calls_missing_no, attributes_upper_limit)
			event["number_of_offers"] = min(offers_no, attributes_upper_limit)
			event["number_of_sent_back"] = min(sent_back, attributes_upper_limit)
			event["W_Fix_incoplete_submission"] = w_fix


	tracefilter_log_pos_5 = attributes_filter.apply_events(tracefilter_log_pos_4, ["TO_REMOVE"],
														 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	xes_exporter.apply(tracefilter_log_pos_5, OUTPUT_PATH)

	for trace in tracefilter_log_pos_5:
		for event in trace:
			if "AZIONE" in event["concept:name"]:
				event["concept:name"] = event["concept:name"].split("AZIONE")[-1]

	xes_exporter.apply(tracefilter_log_pos_5, MID_OUTPUT_PATH)
	print("end")

def addRewardToTraces():
	PATH = os.path.join("..", "final_evaluation", "BPI", "Log", "BPI_2012_log_eng_testing_20.xes")
	OUTPUT_PATH = PATH.replace('.xes', '_preprocessed_number_incr_wfix.xes')

if __name__ == "__main__":
	#preprocessBPI()
	main()
