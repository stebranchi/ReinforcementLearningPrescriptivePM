import pm4py
import itertools

from pm4py.algo.filtering.log.attributes import attributes_filter



def createMDP():
	PATH = '../data/logs/Sepsis Cases - Event Log_training_resources_w_ends.xes'
	OUTPUT_PATH = '../data/mdp/' + PATH.split('/')[-1].replace('.xes', '+resources.csv')
	log = pm4py.read_xes(PATH)
	start_activities = pm4py.get_start_activities(log)
	end_activities = pm4py.get_end_activities(log)
	duration_dict = dict()
	resources_per_event = dict()
	tracefilter_log_pos = attributes_filter.apply_events(log, ["COMPLETE", "complete"],
														 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition", attributes_filter.Parameters.POSITIVE: True})

	#WORKING VERSION
	dfg_dict = dict()
	for trace in tracefilter_log_pos:
		event_name = getEventResourceName(trace[0])
		for event in trace:
			if event['Activity'] != 'START':
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
			if event['Activity'] != 'START' and event['Activity'] != 'END':
				if event['lifecycle:transition'] in ['COMPLETE', 'complete']:
					duration_dict[(event_name, getEventResourceName(event))]['starts'].append(start)
					start = event['time:timestamp']
					duration_dict[(event_name, getEventResourceName(event))]['ends'].append(start)
					event_name = getEventResourceName(event)

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
		resources_per_event[s1] = set()

	for (s1, s2), c in dfg_dict.items():
		#res = s2.split('<')[-1].split("-")[1:].replace(">", "").rstrip()
		# For simple model
		res = "-".join(s2.split('<')[-1].split("-")[1:]).replace(">", "").rstrip()
		if res is not "":
			resources_per_event[s1].add(res)

	new_dfg = dict()
	for (s1, s2), c in dfg_dict.items():
		if s2 == "<END>":
			new_dfg[(s1, s2)] = c
		else:
			#res_s2 = s2.split('<')[-1].split("-")[1].replace(">", "").rstrip()
			# for simple model
			res_s2 = "-".join(s2.split('<')[-1].split("-")[1:]).replace(">", "").rstrip()
			for L in range(0, len(resources_per_event[s1])+1):
				for subset in itertools.combinations(resources_per_event[s1], L):
					subset = list(subset)
					subset.sort()
					if res_s2 in subset:
						s1_key = s1 + "-" + "".join(subset)
						new_dfg[(s1_key, s2)] = c

	for (s1, s2) in new_dfg.keys():
		csv_dict[s1] = []

	for (s1, s2), c in new_dfg.items():
		a = s2.split('<')[-1].split("-")[0].replace(">", "").rstrip()
		if 'END' in s2:
			csv_dict[s1].append((a, s2, c, 0.0))
		else:
			csv_dict[s1].append((a, s2, c, -duration_dict[(s1.split(">-")[0]+">", s2)]['duration']))


	for key, item in csv_dict.items():
		for t in item:
			if (key, t[0]) in max_count.keys():
				max_count[(key, t[0])] += t[2]
			else:
				max_count[(key, t[0])] = t[2]

	for key, item in csv_dict.items():
		for t in item:
			if key.startswith('<START>'):
				starts.append([key, t[0], t[1], t[2]/max_count[(key, t[0])], t[3]])
			else:
				csv.append([key, t[0], t[1], t[2]/max_count[(key, t[0])], t[3]])

	csv = sorted(csv)
	starts = sorted(starts)
	out = open(OUTPUT_PATH, 'w')
	out.write("s, a, s', p(s'|s.a),duration\n<>,START,<START>,1.0,0.0\n")
	for row in starts:
		out.write(','.join([str(x) for x in row]) + '\n')
	for row in csv:
		out.write(','.join([str(x) for x in row]) + '\n')
	out.close()


def getEventResourceName(event):
	if event["Activity"] == 'START' or event["Activity"] == 'START':
		return "<" + event["Activity"] + ">"
	if "org:resource" in event.keys():
		return "<" + event["Activity"] + " - " + str(event['org:resource']) + ">"
	elif "org:group" in event.keys():
		return "<" + event["Activity"] + " - " + str(event['org:group']) + ">"
	else:
		return "<" + event["Activity"] + ">"

	#For simple model
	#return "<" + event["Activity"] + ">"


if __name__ == "__main__":
	createMDP()