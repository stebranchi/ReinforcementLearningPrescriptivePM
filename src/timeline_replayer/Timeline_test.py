import datetime
import random
import os
import numpy as np
import pandas as pd

from queue import PriorityQueue
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py

def main():
	reward_file = os.path.join("..", "..", "cluster_data", "output_mdps", "BPI2012_log_eng_clusters100_squashed_testing_20_preprocessed.csv")
	policy_file = os.path.join("..", "..", "cluster_data", "output_policies", "BPI2012_log_eng_clusters100_squashed_training_80_policy_discount.csv")
	#time_delta_file = os.path.join("..", "data", "BPI2013", "BPI_2012_test_trace_time_delta.csv")
	#with open(time_delta_file) as f:
	#	time_deltas = [float(x) for x in f]
	random_policy = False

	# first and last state definition
	first_state_list = ['<>']     #  first state with new_SB simple model
	# first_state_list = ['<ER Registration-A>', '<ER Registration-L>']
	last_state = '<END>'

	MDP_reward_val = pd.read_csv(reward_file).values
	MDP_policy_val = pd.read_csv(policy_file).values

	#Q-matrix and co.
	states_list, action_dict, reward_dict, resources_availability = get_rules(MDP_policy_val, MDP_reward_val, random_policy)
	completed = list()
	runs = dict()

	if random_policy:
		print("Testing random policy")
	else:
		print("Testing input policy:", policy_file)

	timeline = PriorityQueue()
	last = 0.0
	for i in range(1000):
		#timeline.put_nowait((i*22138, ("<START>", i*22138)))
		value = np.random.randint(0, 200)
		timeline.put_nowait((last, ("<START>", last, i)))
		runs[i] = list()
		runs[i].append(("<START>", last))
		last += value
	cycle = 0
	while not timeline.empty():
		if cycle % 100 == 0:
			print(cycle)
			#print(timeline.queue)
		cycle += 1
		(timestamp, (state, start, i)) = timeline.get()

		if random_policy:
			#exploration: action is taken randomly
			#action, max_val = random.choice(action_dict[state])
			actions = action_dict[state]
			random.shuffle(actions)
		else:
			#max_val = max([value for action, value in action_dict[state]])
			#action = random.choice([k for k, v in action_dict[state] if v == max_val])
			mean = sum([value for action, value in action_dict[state]]) / len(action_dict[state])
			actions = sorted([(action, value) for action, value in action_dict[state] if value >= mean], key=lambda x: x[1], reverse=True)

		summed_probability = 0
		next_state_probability = 0
		choice = random.uniform(0, 1)
		next_state = state
		# stochastic decision
		if state != "<START>":
			resource = state.split("-")[-1].split(">")[0]
			resources_availability[resource] = False
		found = False
		for action, _ in actions:
			if found:
				break
			list_of_possibilities = reward_dict[state][action]
			for possible_next_state in list_of_possibilities.keys():
				next_state_probability = list_of_possibilities[possible_next_state][0]
				if "-" in possible_next_state:
					possible_next_resource = possible_next_state.split("-")[-1].split(">")[0]
					possible_next_resource_available = resources_availability[possible_next_resource]
				else:
					possible_next_resource_available = False
				if summed_probability < choice and choice <= summed_probability + next_state_probability and possible_next_resource_available is False:
					next_state = possible_next_state
					next_resource = possible_next_resource
					reward = list_of_possibilities[next_state][1]
					found = True
				summed_probability = summed_probability + next_state_probability

		if '<END>' in next_state:
			completed.append((next_state, timestamp - start))
			runs[i].append((next_state, timestamp - reward))
		else:
			if next_state == state:
				timestamp = timestamp + 200
			else:
				timestamp = timestamp - reward
				resources_availability[next_resource] = True
			state = next_state
			timeline.put_nowait((timestamp, (state, start, i)))
			runs[i].append((next_state, timestamp - reward))

	mean_duration = 0
	for item in completed:
		mean_duration += item[1]

	create_log(runs, "../data/generated_logs/BPI_filtered_old.xes")
	print(completed)
	print("mean duration: " + str(mean_duration/len(completed)))


def get_rules(MDP_policy_val, MDP_reward_val, random_policy):
	#extract states
	policy_states_list = np.unique(np.transpose(MDP_policy_val)[0])
	reward_states_list = np.unique(np.transpose(MDP_reward_val)[0])
	resources = list()
	resources_availabity = dict()

	if random_policy:
		states_list = reward_states_list
	else:
		states_list = [state for state in policy_states_list if state in reward_states_list] # list of states available

	# define action dictionary and q_table
	policy_action_dict = {} # key = state : value = tuple(action, q-value)
	reward_action_dict = {} # key = state : value = action
	reward_dict = {} # key1 = state : value1 = dict( key2 = action : value2 = dict( key3 = next_state : value3 = list(probability, reward) ) )
	for s in states_list:
		policy_action_dict[s] = []  # list
		reward_action_dict[s] = []  # list
		reward_dict[s] = {}
		if len(s.split("-")) > 1:
			resource = s.split("-")[-1].split(">")[0]
			resources.append(resource)

	for r in set(resources):
		resources_availabity[r] = False

	for rowid, row in enumerate(MDP_policy_val):
		state = MDP_policy_val[rowid, 0]
		state_action = MDP_policy_val[rowid, 1]
		action_q_value = MDP_policy_val[rowid, 5]
		if state in states_list:
			if state_action not in [action for action, q_value in policy_action_dict[state]]:
				policy_action_dict[state].append((state_action, action_q_value))

	for rowid, row in enumerate(MDP_reward_val):
		state = MDP_reward_val[rowid, 0]
		state_action = MDP_reward_val[rowid, 1]
		if state in states_list:
			if state_action not in reward_action_dict[state]:
				reward_action_dict[state].append(state_action)
				reward_dict[state][state_action] = {}
				# q_table[state][state_action] = MDP_policy_val[rowid, 5]
			next_state = MDP_reward_val[rowid, 2]
			next_state_probability = MDP_reward_val[rowid, 3]
			next_state_reward = MDP_reward_val[rowid, 4]
			reward_dict[state][state_action][next_state] = [next_state_probability, next_state_reward]

	if random_policy:
		states_list = reward_states_list
		action_dict = {state: [(action, 0) for action in reward_action_dict[state]] for state in reward_states_list} # in this case condider all actions of reward model
	else:
		action_dict = {state: [(action, q_value) for action, q_value in policy_action_dict[state] if action in reward_action_dict[state]] # in this case consider action in common between reward and policy model
					   for state in states_list}

	return states_list, action_dict, reward_dict, resources_availabity


def create_log(runs, path):
	today = datetime.datetime.today()
	log = EventLog()
	for i, run in runs.items():
		trace = Trace()
		for (action, timestamp) in run:
			event = Event()
			event["concept:name"] = action.split("<")[-1].replace(">", "")
			days_to_add = timestamp / 28800
			seconds_to_add = timestamp % 28800
			new_date = today + datetime.timedelta(days=days_to_add) + datetime.timedelta(seconds=seconds_to_add)
			if new_date.time().hour > 18:
				new_date = new_date + datetime.timedelta(days=days_to_add) - datetime.timedelta(hours=8)
			event["time:timestamp"] = new_date
			trace.append(event)
		log.append(trace)
	xes_exporter.apply(log, path)


if __name__ == "__main__":
	main()