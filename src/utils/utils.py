import os
import numpy as np
import pm4py
import datetime
from MDPCreator_BPI import getEventResourceName
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import Event


def remove_q():
	policy_path = os.path.join("..", "data", "output", "new_tests", "BPI_2012_log_eng_preprocessed_filtered_reward.csv")
	output_path = policy_path.replace(".csv", "_no_q.csv")
	policy_file = open(policy_path, 'r')
	output_file = open(output_path, 'w')
	for line in policy_file:
		output_file.write(",".join(line.split(",")[:-1]) + "\n")

	output_file.close()
	policy_file.close()


def update_policy():
	reward_path = os.path.join("..", "data", "mdp", "BPI_2012_MDP_r.csv")
	output_path = reward_path.replace(".csv", "_distribution.csv")
	reward_file = open(reward_path, 'r')
	output_file = open(output_path, 'w')
	first = True
	for line in reward_file:
		values = line.rstrip().split(",")
		if first:
			first = False
			values[-1] = "distribution_reward"
			values.append("mean")
			values.append("deviation")
			output_file.write(",".join(values) + "\n")
		else:
			rew = float(values[4])
			distribution = np.random.randn(1)
			if distribution > 0.7:
				values[-1] = "exponential"
				values.append(str(-1*rew))
				values.append("0")
			else:
				values[-1] = "Normal"
				values.append(str(-1*rew))
				deviation = float(str(rew).rstrip()) * -1 / 5
				values.append(str(deviation))

			output_file.write(",".join(values) + "\n")

	output_file.close()
	reward_file.close()


def zero_q():
	policy_path = os.path.join("..", "data", "mdp", "BPI_2012_MDP_r3_Q_3.csv")
	output_path = policy_path.replace(".csv", "_zero_q.csv")
	policy_file = open(policy_path, 'r')
	output_file = open(output_path, 'w')
	for line in policy_file:
		values = line.split(",")
		values[-1] = '0'
		output_file.write(",".join(values) + "\n")

	output_file.close()
	policy_file.close()


def parsemdp():
	policy_path = os.path.join("..", "data", "mdp", "Trimmed_BPI_2012_model.csv")
	output_path = policy_path.replace(".csv", "_no_shift.csv")
	policy_file = open(policy_path, 'r')
	output_file = open(output_path, 'w')
	first = True
	for line in policy_file:
		if first:
			first = False
			output_file.write(line)
		else:
			values = line.split(",")
			if float(values[-1]) < 0.0:
				values[-1] = str(float(values[-1]) + 10000.0)
			output_file.write(",".join([x.strip() for x in values]) + "\n")

	output_file.close()
	policy_file.close()


def find_event():
	policy_path = os.path.join("..", "data", "BPI2013", "BPI_2012_log_eng_preprocessed_filtered_v3.xes")
	tracefilter_log_pos = pm4py.read_xes(policy_path)
	for trace in tracefilter_log_pos:
		event1 = False
		event2 = False
		for event in trace:
			if getEventResourceName(event) == "<O_SENT - high - 0 - 0 - 0>":
				event1 = True
			if getEventResourceName(event) == "<W_Call_after_offerAZIONEA_CANCELLED - high - 2 - 0 - 1>":
				event2 = True
			if event1 and event2:
				print(trace.attributes["concept:name"])


def splitLog():
	policy_path = os.path.join("..", "old_final_evaluation", "Fine", "log", "Road_Traffic_Fine_Management_Process_filter_dismissal.xes")
	tracefilter_log_pos = pm4py.read_xes(policy_path)
	output_training = os.path.join("..", "final_evaluation", "Fine", "Log", "Road_Traffic_Fine_training_80.xes")
	output_testing = os.path.join("..", "final_evaluation", "Fine", "Log", "Road_Traffic_Fine_testing_20.xes")

	traces_list = [x for x in tracefilter_log_pos]
	train_l = int(len(traces_list) / 100 * 80)
	train_log = EventLog()
	test_log = EventLog()

	for i, t in enumerate(traces_list):
		if i <= train_l:
			train_log.append(t)
		else:
			test_log.append(t)

	xes_exporter.apply(train_log, output_training)
	xes_exporter.apply(test_log, output_testing)


def count():
	PATH = os.path.join("..", "final_evaluation", "BPI", "Log", "BPI_2012_log_eng_testing_40_mid_preprocessed.xes")
	log = pm4py.read_xes(PATH)
	summ = 0
	c = 0
	for trace in log:
		for event in trace:
			if "duration" in event.keys():
				summ += event["duration"]
				c += 1

	print(str(summ/c))

def scale_reward():
	PATH_TRIMMED = os.path.join("..", "final_evaluation", "BPI", "Mdp", "Trimmed BPI_2012 mdp training 60 var90 r10.csv")
	PATH_MDP = os.path.join("..", "final_evaluation", "BPI", "Mdp", "BPI_2012_log_eng_training_60_preprocessed_var90_1.csv")
	PATH_OUTPUT = os.path.join("..", "final_evaluation", "BPI", "Mdp", "Trimmed BPI_2012 mdp training 60 var90 scaled3 r10.csv")
	csv_trimmed = open(PATH_TRIMMED, "r")
	csv = open(PATH_MDP, "r")
	output = open(PATH_OUTPUT, 'w')

	for line in csv_trimmed:
		parts = line.split(",")
		parts[-1].rstrip()
		if "O_ACCEPTED" in parts[2]:
			with open(PATH_MDP) as csv:
				for l in csv:
					pts = l.split(",")
					if pts[0] == parts[0] and pts[1] == parts[1] and pts[2] == parts[2]:
						occ = pts[-1].rstrip()
				rew = (int(occ)/3)**2/(1+(int(occ)/3)**2) * int(parts[-1])
				parts[-1] = str(rew) + "\n"
		output.write(",".join([str(x) for x in parts]))

	csv_trimmed.close()
	csv.close()
	output.close()


def addDurationReward(path):
	log = pm4py.read_xes(path)
	output_path = path.replace(".xes", "_rewards_and_durations.xes")
	begin_event = Event()
	begin_event["concept:name"] = ""

	tracefilter_log = attributes_filter.apply_events(log, ["SCHEDULE"],
														   parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition", attributes_filter.Parameters.POSITIVE: False})

	for trace in tracefilter_log:
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
			if event["concept:name"] not in ["START", "END"]:
				if event["concept:name"].startswith("W_") and event["lifecycle:transition"] == "START":
					begin_event = event
				if event["concept:name"] == begin_event["concept:name"] and event["lifecycle:transition"] == "COMPLETE":
					#duration = datetime.datetime.strptime(str(event["time:timestamp"]), "%Y-%m-%dT%H:%M:%S.%f%z") - datetime.datetime.strptime(str(begin_event["time:timestamp"]), "%Y-%m-%dT%H:%M:%S.%f%z")
					duration = event["time:timestamp"] - begin_event["time:timestamp"]
					# controllare due giorni diversi invece che durata
					#if datetime.datetime.strptime(str(event["time:timestamp"]), "%Y-%m-%dT%H:%M:%S.%f%z").day is not datetime.datetime.strptime(str(begin_event["time:timestamp"]), "%Y-%m-%dT%H:%M:%S.%f%z").day:
					if event["time:timestamp"].day is not begin_event["time:timestamp"].day:
						begin_event["duration"] = 2000
					else:
						begin_event["duration"] = duration.total_seconds()
					begin_event = Event()
					begin_event["concept:name"] = ""
				if last_event["concept:name"] == event["concept:name"] and event["concept:name"] is not "TO_REMOVE":
					if "duration" in last_event.keys() and "duration" in event.keys():
						event["duration"] = event["duration"] + last_event["duration"]
					idx = [x for x, e in enumerate(trace) if e == last_event][0]
					trace[idx]["concept:name"] = "TO_REMOVE"
			last_event = event

	tracefilter_log_pos_2 = attributes_filter.apply_events(tracefilter_log, ["TO_REMOVE"],
														   parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	for trace in tracefilter_log_pos_2:
		if "AMOUNT_REQ" in trace.attributes.keys():
			amount = int(trace.attributes["AMOUNT_REQ"])
		else:
			amount = 0
		for event in trace:
			reward = 0
			if 'duration' in event.keys():
				reward -= 0.005 * event['duration']
			if event["concept:name"] == "A_ACCEPTED":
				reward += 0.15 * amount
			event["kpi:reward"] = reward

	xes_exporter.apply(tracefilter_log_pos_2, output_path)

if __name__ == '__main__':
	#remove_q()
	#update_policy()
	#zero_q()
	#parsemdp()
	#find_event()
	#splitLog()
	#count()
	#scale_reward()
	addDurationReward("../data/BPI2013/BPI_2012_log_eng.xes")