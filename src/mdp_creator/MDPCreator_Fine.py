import pm4py
import os
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import Event
import copy

def main(log):
	PATH = os.path.join("..", "final_evaluation", "Fine", "Log", "Road_Traffic_Fine_testing_40_preprocessed_r3.xes")
	OUTPUT_PATH = PATH.replace("Log", "Mdp").replace('.xes', '.csv')
	#tracefilter_log_pos = pm4py.read_xes(PATH)

	tracefilter_log_pos = log
	duration_dict = dict()

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
				duration_dict[(event_name, new_event_name)]['starts'].append(start)
				start = event['time:timestamp']
				duration_dict[(event_name, new_event_name)]['ends'].append(start)
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
			a = parts[0].replace("<", "").replace(">", "")
		else:
			s2_csv = s2
			a = s2.split(" - ")[0].replace("<", "").replace(">", "").rstrip()
		if "Payment full" in s2 and "Complete" not in s2:
			two_months = int(s2.split(" - ")[0].split("-")[-1].strip())
			if two_months < 3:
				reward = 3
			elif two_months < 6:
				reward = 2
			else:
				reward = 1
			#reward = 6 - int(s2.split(" - ")[0].split("-")[-1].strip())
		if ("Send Appeal to Prefecture-#" in s2_csv or "Appeal to Judge-G" in s2_csv) and "Complete" not in s2_csv:
			reward = -1
		if 'END' in s2:
			if (a, s2_csv) in csv_dict[s1_csv].keys():
				csv_dict[s1_csv][(a, s2_csv)]["c"] = csv_dict[s1_csv][(a, s2_csv)]["c"] + c
			else:
				csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": 0.0}
		else:
			if (a, s2_csv) in csv_dict[s1_csv].keys():
				csv_dict[s1_csv][(a, s2_csv)]["c"] = csv_dict[s1_csv][(a, s2_csv)]["c"] + c
			else:
				csv_dict[s1_csv][(a, s2_csv)] = {"c": c, "rew": reward}

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
	out.write("s, a, s', p_r, duration, occurences\n<>,START,<START>,1.0,0.0,0\n")
	for row in starts:
		out.write(','.join([str(x) for x in row]) + '\n')
	for row in csv:
		out.write(','.join([str(x) for x in row]) + '\n')
	out.close()

#v1
def getEventResourceName(event):
	if event["concept:name"] == 'START' or event["concept:name"] == 'END':
		return "<" + event["concept:name"] + ">"
	if "two_months_passed" in event.keys():
		return "<" + event["concept:name"] + " - " + str(event["amount_category"]) + ">"
	else:
		return "<" + event["concept:name"] + "-0 - " + str(event["amount_category"]) + ">"
#v2
def getEventResourceNamev2(event):
	if event["concept:name"] == 'START' or event["concept:name"] == 'END':
		return "<" + event["concept:name"] + ">"
	if "two_months_passed" in event.keys():
		return "<" + event["concept:name"] + " - " + str(event["amount_category"]) + " - " + str(event["b1"]) + " - " + str(event["b2"]) + " - " + str(event["b3"]) + " - " + str(event["b4"]) + ">"
	else:
		return "<" + event["concept:name"] + " - " + str(event["amount_category"]) + str(event["b1"]) + " - " + str(event["b2"]) + " - " + str(event["b3"]) + " - " + str(event["b4"]) + ">"



def preprocessFine():
	PATH = os.path.join("..", "final_evaluation", "Fine", "Log", "Road_Traffic_Fine_testing_40.xes")
	OUTPUT_PATH = PATH.replace('.xes', '_preprocessed_r3.xes')
	MID_OUTPUT_PATH = PATH.replace('.xes', '_mid_preprocessed_r3.xes')
	attributes_upper_limit = 7
	log = pm4py.read_xes(PATH)
	begin_event = Event()
	begin_event["concept:name"] = ""

	#filtered_log = variants_filter.filter_log_variants_percentage(log, percentage=0.9)

	for trace in log:
		start_time = None
		expense = 0
		amount = 0
		(b1, b2, b3, b4) = (False, False, False, False)
		payment_idx = []
		for ev in trace:
			if "expense" in ev.keys():
				expense = ev["expense"]
			if "amount" in ev.keys():
				amount = ev["amount"]
				if amount < 50:
					amount_cat = "low"
				elif amount >= 50:
					amount_cat = "high"
			ev["amount_category"] = amount_cat
			if ev["concept:name"] == "Create Fine":
				start_time = ev["time:timestamp"]
			if start_time:
				two_months_passed = (ev["time:timestamp"] - start_time).days//60
				ev["two_months_passed"] = min(two_months_passed, attributes_upper_limit)
			if ev["concept:name"] == "Payment":
				if ev["totalPaymentAmount"] >= (amount + expense):
					ev["concept:name"] = "Payment full fine"
				else:
					ev["concept:name"] = "Payment partly"
				idx = [x for x, e in enumerate(trace) if e == ev][0]
				payment_idx.append(idx)
			if ev["concept:name"].split("-")[0] in ["Insert Date Appeal to Prefecture", "Notify Result Appeal to Offender", "Receive Result Appeal from Prefecture", "Insert Fine Notification"]:
				ev["concept:name"] = "TO_REMOVE"
			if ev["concept:name"].split("-")[0] in ["Send Appeal to Prefecture", "Appeal to Judge"]:
				ev["concept:name"] = ev["concept:name"] + "-" + ev["dismissal"]
			if "two_months_passed" in ev.keys() and ev["concept:name"] is not "TO_REMOVE":
				ev["concept:name"] += "-" + str(ev["two_months_passed"])

		for idx in payment_idx[:-1]:
			trace[idx]["concept:name"] = "TO_REMOVE"
		start = Event()
		start["concept:name"] = "START"
		start["time:timestamp"] = trace[0]["time:timestamp"]
		trace.insert(0, start)
		end = Event()
		end["concept:name"] = "END"
		end["time:timestamp"] = trace[-1]["time:timestamp"]
		trace.append(end)

	tracefilter_log_pos = attributes_filter.apply_events(log, ["TO_REMOVE"],
														   parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	for trace in tracefilter_log_pos:
		trace_id = trace.attributes["concept:name"]
		for event in trace:
			if ("Send Appeal to Prefecture" in event["concept:name"] or "Appeal to Judge" in event["concept:name"] or "Payment" in event["concept:name"]) and "Complete" not in event["concept:name"]:
				idx = [x for x, e in enumerate(trace) if e == event][0]
				ev = Event()
				parts = event["concept:name"].split("-")
				if len(parts) > 1:
					name = "-".join(parts[:-1]) + " Complete-" + parts[-1]
				ev["concept:name"] = name
				ev["time:timestamp"] = event["time:timestamp"]
				ev["amount_category"] = event["amount_category"]
				if "two_months_passed" in event.keys():
					ev["two_months_passed"] = event["two_months_passed"]
					"""ev["b1"] = event["b1"]
					ev["b2"] = event["b2"]
					ev["b3"] = event["b3"]
					ev["b4"] = event["b4"]"""
				trace.insert(idx+1, ev)
				event["concept:name"] = trace[idx-1]["concept:name"] + "AZIONE" + event["concept:name"]
				#event["action"] = trace[idx-1]["concept:name"]
				trace[idx-1]["concept:name"] = "TO_REMOVE"

	tracefilter_log_pos2 = attributes_filter.apply_events(tracefilter_log_pos, ["TO_REMOVE"],
														 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	xes_exporter.apply(tracefilter_log_pos2, OUTPUT_PATH)

	main(tracefilter_log_pos2)

	for trace in tracefilter_log_pos2:
		for event in trace:
			if "AZIONE" in event["concept:name"]:
				event["concept:name"] = event["concept:name"].split("AZIONE")[-1]

	xes_exporter.apply(tracefilter_log_pos2, MID_OUTPUT_PATH)
	print("end")

if __name__ == "__main__":
	preprocessFine()
	#main()
