import pm4py
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.conversion.dfg import converter as dfg_mining
from pm4py.algo.filtering.log.attributes import attributes_filter

def main():
	PATH = '../data/logs/simple_model+changes_training_res_complete_no_res.xes'
	OUTPUT_PATH = '../data/mdp/' + PATH.split('/')[-1].replace('.xes', 'test.csv')
	tracefilter_log_pos = pm4py.read_xes(PATH)
	#start_activities = pm4py.get_start_activities(log)
	#end_activities = pm4py.get_end_activities(log)
	duration_dict = dict()
	resources_per_event = dict()
	#tracefilter_log_pos = attributes_filter.apply_events(log, ["COMPLETE", "complete"],
	#													 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition", attributes_filter.Parameters.POSITIVE: True})

	#WORKING VERSION
	dfg_dict = dict()
	for trace in tracefilter_log_pos:
		event_name = getEventResourceName(trace[0])
		for event in trace:
			if event['concept:name'] != 'START':
				key = (event_name, getEventResourceName(event))
				duration_dict[(event_name, getEventResourceName(event))] = {'starts': list(), 'ends': list(), 'count': 0, 'sum': 0, 'duration': 0}
				event_name = getEventResourceName(event)
				if key in dfg_dict.keys():
					dfg_dict[key] += 1
				else:
					dfg_dict[key] = 1

	for trace in tracefilter_log_pos:
		start = trace[0]['time:timestamp']
		event_name = getEventResourceName(trace[0])
		for event in trace:
			if event['concept:name'] != 'START' and event['concept:name'] != 'END':
				if event['lifecycle:transition'] == 'COMPLETE':
					duration_dict[(event_name, getEventResourceName(event))]['starts'].append(start)
					start = event['time:timestamp']
					duration_dict[(event_name, getEventResourceName(event))]['ends'].append(start)
					event_name = getEventResourceName(event)

	#MDP with full prefix
	"""dfg_dict = dict()

	for trace in tracefilter_log_pos:
		event_name = [getEventResourceName(trace[0])]
		for event in trace:
			if event['concept:name'] != 'START':
				event_name.append(getEventResourceName(event))
				s1 = str.join("|", [x for x in event_name[:-1]])
				s2 = str.join("|", [x for x in event_name])
				key = (s1, s2)
				duration_dict[key] = {'starts': list(), 'ends': list(), 'count': 0, 'sum': 0, 'duration': 0}
				if key in dfg_dict.keys():
					dfg_dict[key] += 1
				else:
					dfg_dict[key] = 1

	for trace in tracefilter_log_pos:
		start = trace[0]['time:timestamp']
		event_name = [getEventResourceName(trace[0])]
		for event in trace:
			if event['concept:name'] != 'START':
				if event['lifecycle:transition'] == 'complete':
					event_name.append(getEventResourceName(event))
					s1 = str.join("|", [x for x in event_name[:-1]])
					s2 = str.join("|", [x for x in event_name])
					duration_dict[(s1, s2)]['starts'].append(start)
					start = event['time:timestamp']
					duration_dict[(s1, s2)]['ends'].append(start)
	"""

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
		csv_dict[s1] = []
		resources_per_event[s1] = set()

	for (s1, s2), c in dfg_dict.items():
		a = s2.split('<')[-1].split("-")[0].replace(">", "").rstrip()
		res = s2.split('<')[-1].split("-")[-1].replace(">", "").rstrip()
		if res is not "":
			resources_per_event[s1].add(res)
		if 'END' in s2:
			csv_dict[s1].append((a, s2, c, 0.0))
		else:
			csv_dict[s1].append((a, s2, c, -duration_dict[(s1, s2)]['duration']))

	for key, item in csv_dict.items():
		for t in item:
			if (key, t[0]) in max_count.keys():
				max_count[(key, t[0])] += t[2]
			else:
				max_count[(key, t[0])] = t[2]

	for key, item in csv_dict.items():
		for t in item:
			tmp = [key, t[0], t[1], t[2]/max_count[(key, t[0])], t[3], max_count[(key, t[0])]/sum([v for (k1, _), v in max_count.items() if k1 == key])]
			if key == '<START>':
				starts.append(tmp)
			else:
				csv.append(tmp)

	csv = sorted(csv)
	starts = sorted(starts)
	out = open(OUTPUT_PATH, 'w')
	out.write("s,a,s',p_r,duration,p_a\n<>,START,<START>,1.0,0.0,1.0\n")
	for row in starts:
		out.write(','.join([str(x) for x in row]) + '\n')
	for row in csv:
		out.write(','.join([str(x) for x in row]) + '\n')
	out.close()


def getEventResourceName(event):
	if event["concept:name"] == 'START' or event["concept:name"] == 'START':
		return "<" + event["concept:name"] + ">"
	if "org:resource" in event.keys():
		return "<" + event["concept:name"] + " - " + str(event['org:resource']) + ">"
	elif "org:group" in event.keys():
		return "<" + event["concept:name"] + " - " + str(event['org:group']) + ">"
	else:
		return "<" + event["concept:name"] + ">"


if __name__ == "__main__":
	main()