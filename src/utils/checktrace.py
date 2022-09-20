from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.log import log as lg
from datetime import datetime, timedelta
from pm4py.objects.petri_net.importer import importer as pnml_importer


def checkTrace():
	possible_actions = ["A", "B", "C", "D", "E", "F", "G", "H"]
	state = [0, 1, 4, 6, 7, 8]
	net, initial_marking, final_marking = pnml_importer.apply("../xes/simple_example2.pnml")
	reward = 0
	for i in range(len(state)):
		log= lg.EventLog()
		trace = lg.Trace()
		trace.attributes["concept:name"] = 1
		c = 1
		end = False
		event = lg.Event()
		event["concept:name"] = "START"
		event["time:timestamp"] = (datetime.now()).timestamp()
		trace.append(event)
		for action in range(len(state)):
			event = lg.Event()
			if not end:
				if state[action] == 8:
					event["concept:name"] = "END"
					end = True
				else:
					event["concept:name"] = possible_actions[int(state[action])]
				event["time:timestamp"] = (datetime.now() + timedelta(hours=c)).timestamp()
				c += 1
			trace.append(event)
		log.append(trace)
		replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)
		checker = replayed_traces[0]
		reward += len(checker['activated_transitions']) * 3 - len(checker['transitions_with_problems'])
	print(reward)


if __name__ == '__main__':
	checkTrace()