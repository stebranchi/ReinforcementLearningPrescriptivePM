import random
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from queue import PriorityQueue

def main():
	# with resources management!!!!!!!!!!
	# fixed a bug for states without resources in get_resource(state)
	execution_start = datetime.now()

	reward_file = os.path.join("..", "input_data", "sepsis", "model", "sepsis_model_MDP_r_test.csv")
	policy_file = os.path.join("..", "output_data", "sepsis", "Q_values", "sepsis_model_MDP_r_decl3_Q.csv")
	# policy_file = os.path.join("..", "output_data", "sepsis", "results_stefano", "sepsis_model_MDP_r_decl_MC_full_constraint.csv")

	# reward_file = os.path.join("..", "input_data", "BPI_2012", "model", "BPI_2012_MDP_r.csv")
	# policy_file = os.path.join("..", "output_data", "BPI_2012", "Q_values", "BPI_2012_MDP_r3_Q_3.csv")

	wait_time_file = os.path.join("..", "output_data", "sepsis", "trace_time_delta.csv")
	with open(wait_time_file, "r") as f:
		wait_time_list = [float(item.strip()) for item in f]


	random_policy = False
	busy_resource_strategy = 'subopt'  # 'wait' it waits until resource is freed, 'subopt' it selects suboptimal action

	# first and last state definition
	# first_state_list = ['<>']     #  first state with new_SB simple model
	# first_state_list = ['<ER Registration-A>', '<ER Registration-L>']
	# last_state = '<END>'

	MDP_reward_val = pd.read_csv(reward_file).values
	MDP_policy_val = pd.read_csv(policy_file).values

	#Q-matrix and co.
	states_list, action_dict, reward_dict, resources_unavailability = get_rules(MDP_policy_val, MDP_reward_val, random_policy)
	completed = list()

	if random_policy:
		print("Testing random policy")
	else:
		print("Testing input policy:", policy_file)

	timeline = PriorityQueue()
	number_of_runs = 100000
	for i in range(number_of_runs):
		# wait_time = 20000 # constant
		wait_time = random.choice(wait_time_list)
		timeline.put_nowait((i * wait_time, ("<START>", i * wait_time)))


	# cycle = 0
	last_check_datetime = datetime.now()
	while not timeline.empty():
		# # these 5 lines are running test
		# if cycle%100 == 0:
		# 	print(cycle)
		# # 		#	print(timeline.queue)
		# cycle += 1

		(timestamp, (state, start)) = timeline.get()

		if random_policy:
			#exploration: action is taken randomly
			#action, max_val = random.choice(action_dict[state]) # old code
			actions = action_dict[state]
			random.shuffle(actions) # all the actions shuffled
		else:
			# old code start
			#max_val = max([value for action, value in action_dict[state]]) # old code
			#action = random.choice([k for k, v in action_dict[state] if v == max_val])
			# old code end
			if busy_resource_strategy == 'subopt':
				# selects all actions better on equal than the mean and orders them from best to worst
				mean = sum([value for action, value in action_dict[state]]) / len(action_dict[state])
				actions = sorted([(action, value) for action, value in action_dict[state] if value >= mean],
								 key=lambda x: x[1], reverse=True)
			elif busy_resource_strategy == 'wait':
				# selects all actions better on equal than the best one
				best = max([value for action, value in action_dict[state]])
				actions = sorted([(action, value) for action, value in action_dict[state] if value >= best],
								 key=lambda x: x[1], reverse=True)

		summed_probability = 0
		next_state_probability = 0
		choice = random.uniform(0, 1)
		next_state = state
		# stochastic decision
		if state != "<START>":
			resource = get_resource(state)
			resources_unavailability[resource] = False # False = available # it frees the resource
		found = False
		for action, _ in actions:
			if found:
				break
			list_of_possibilities = reward_dict[state][action]
			for possible_next_state in list_of_possibilities.keys():
				next_state_probability = list_of_possibilities[possible_next_state][0]
				possible_next_resource = get_resource(possible_next_state)
				if summed_probability < choice and choice <= summed_probability + next_state_probability and \
						(resources_unavailability[possible_next_resource] is False or possible_next_resource == ""):
					next_state = possible_next_state
					next_resource = possible_next_resource
					reward = list_of_possibilities[next_state][1]
					found = True
				summed_probability = summed_probability + next_state_probability

		if '<END>' in next_state:
			completed.append((next_state, timestamp - start))
			if len(completed) % 1000 == 0:
				print(len(completed))
			# debug check
			# if len(completed) == round(number_of_runs/2):
			# 	test_file = os.path.join("..", "output_data", "BPI_2012", "resource_busy_test.csv")
			# 	with open(test_file, 'w') as f:
			# 		for k, v in resources_unavailability.items():
			# 			item = k + ',' + str(v)
			# 			print(item)
			# 			f.write("%s\n" % item)
		else:
			if next_state == state:
				timestamp = timestamp + 900
				# non potrebbe succedere che nel frattempo un'altra traccia gli frega di nuovo la risorsa e quindi i tempi medi si dilatano moltissimo?
				# forse questo non è un problema
			else:
				timestamp = timestamp - reward
				resources_unavailability[next_resource] = True # True = unavailable
			state = next_state
			timeline.put_nowait((timestamp, (state, start)))

		# if datetime.now() > last_check_datetime + timedelta(seconds=60):
		# 	last_check_datetime = datetime.now()

	mean_duration = 0
	for item in completed:
		mean_duration += item[1]

	print(completed)
	execution_end = datetime.now()
	execution_time = execution_end - execution_start
	print("mean duration: " + str(mean_duration/len(completed)))
	print("execution_time:", execution_time)


def get_resource(state):
	if len(state.split("-")) > 1: # added this to manage state without resources
		resource = state.split("-")[-1].split(">")[0]
	else:
		resource = ""

	return resource


def get_rules(MDP_policy_val, MDP_reward_val, random_policy):
	#extracts states
	policy_states_list = np.unique(np.transpose(MDP_policy_val)[0])
	reward_states_list = np.unique(np.transpose(MDP_reward_val)[0])
	resources = list()
	resources_unavailabity = dict()

	if random_policy:
		states_list = reward_states_list
	else:
		states_list = [state for state in policy_states_list if state in reward_states_list] # list of states available

	# defines action dictionary and q_table
	policy_action_dict = {} # key = state : value = tuple(action, q-value)
	reward_action_dict = {} # key = state : value = action
	reward_dict = {} # key1 = state : value1 = dict( key2 = action : value2 = dict( key3 = next_state : value3 = list(probability, reward) ) )
	for s in states_list:
		policy_action_dict[s] = []  # list
		reward_action_dict[s] = []  # list
		reward_dict[s] = {}
		resource = get_resource(s)
		resources.append(resource)

	for r in set(resources):
		resources_unavailabity[r] = False # all the resources start as available

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

	return states_list, action_dict, reward_dict, resources_unavailabity


if __name__== "__main__":
	main()