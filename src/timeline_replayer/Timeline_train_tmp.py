import random
import os
import itertools

import numpy as np
import pandas as pd
import time
from queue import PriorityQueue
from IPython.display import clear_output
from pm4py.objects.log import obj as lg

def main():
	# use SARSA instead of Q-learning

	print("Importing the model")
	start_time = time.time()
	#import csv, must have these columns (s,a,s',p,r) these are respectively: state,    action, next state, probability, reward(e.g. -time execution)
	# input_file = os.path.join("..", "input_data", "test", "model", "test_model_1_r.csv")
	# output_file = os.path.join("..", "output_data", "test", "Q_values", "test_model_1_r_Q_MC_decl.csv")

	#input_file = os.path.join("..", "input_data", "sepsis", "model", "sepsis_model_MDP_r.csv")
	#output_file = os.path.join("..", "output_data", "sepsis", "Q_values", "sepsis_model_MDP_r_decl2_Q_bis.csv")

	#input_file = os.path.join("..", "data", "mdp", "sepsis_model_MDP_r.csv")
	#output_file = os.path.join("..", "data", "output", "sepsis_model_MDP_r_MC_90_1_ub.csv")
	#decl_file = os.path.join("..", "data", "declare", "stefano", "Sepsis_training_90_1_unary_binary+.decl")
	#txt_file = os.path.join("..", "data", "declare", "stefano", "Sepsis_training_90_1_unary_binary+.txt")

	input_file = os.path.join("..", "data", "mdp", "simple_model+changes_training+resources.csv")
	output_file = os.path.join("..", "data", "output", "new_tests", "simple_model_states+resources_all_next_resources_no_waittitme.csv")

	# first and last state definition
	# first_state = 'Start' #  first state with manual simple model
	first_state = '<>'      #  first state with new_SB simple model
	# last_state = 'Stop'   #  last state with manual simple model
	last_state = '<END>'    #  last state with new_SB simple model

	# load the policy model
	MDPcsv = pd.read_csv(input_file)

	#convert into array
	MDPval = MDPcsv.values
	header = list(MDPcsv.columns.array)

	#Q-matrix and co.
	states_list, action_lists, q_table, resources_per_event = new_generate_q_table(MDPval)

	# constraint rewards
	declare_reward = 10000
	min_support_constraint = 90

	#RL hyperparameters
	alpha_max = 0.1 # initial learning rate
	alpha_min = 0.0001 # final learning rate
	gamma = 0.9 # discount rate, discount near to 1 avoids loops
	epsilon = 0.1 # discovery rate

	print("Training the agent")

	#loop on episodes
	number_of_runs = 100000
	points_dict = None
	for i in range(1, number_of_runs):
		timeline = PriorityQueue()
		last = 0
		reward = list()
		return_dict = list()
		path = list()
		random_variation = np.random.randint(100)
		for j in range(50 + random_variation):
			#timeline.put_nowait((i*22138, ("<START>", i*22138)))
			value = np.random.normal(20000, 5000)
			timeline.put_nowait((last, ("", "", "<>", last, j)))
			reward.append(0)
			return_dict = {}
			path.append(["<>"])
			last += value

		# linearly variable alpha: alpha=alpha_max when i=1, alpha=alpha_min when i=number_of_runs
		alpha = alpha_max - (alpha_max - alpha_min)*(i-1)/(number_of_runs-1)
		cycle = 0
		while not timeline.empty():
			wait = False
			if cycle % 100 == 0:
				print(cycle)
				#print(timeline.queue)
			cycle += 1
			# these three variables manage stochastic decision
			(timestamp, (old_old_state, old_state, state, start, i)) = timeline.get()
			if " - " in old_state:
				resource = old_state.split(" - ")[-1].split(">")[0]
				resources_per_event[old_old_state.split(">")[0] + ">"][resource] = False
			if state is not "<>":
				state_no_res = state.split(">")[0] + ">"
				possible_res = [x for x in resources_per_event[state_no_res].keys()]
				possible_available_res = [x for x in resources_per_event[state_no_res].keys() if not resources_per_event[state_no_res][x]]
				if len(possible_res) > 0:
					if len(possible_res) > 0:
						state = state_no_res + "- " + " ".join(possible_res)
					else:
						if " - " in state:
							resource = state.split(" - ")[-1].split(">")[0]
							resources_per_event[old_state.split(">")[0] + ">"][resource] = False
						wait = True
			if wait:
				timeline.put_nowait((timestamp + 200, (old_old_state, old_state, state, start, i)))
			else:
				action = select_action(action_lists, q_table, state, epsilon)
				summed_probability = 0
				next_state_probability = 0
				choice = random.uniform(0, 1)
				next_state = 'Undefined'
				# stochastic decision
				found = False
				list_of_possibilities = q_table[state][action][1]
				p = np.array([v[0] for x,v in list_of_possibilities.items()])
				p /= p.sum()
				next_state = np.random.choice([x for x in list_of_possibilities.keys()], p=p)
				reward = list_of_possibilities[next_state][1]

				next_state_res = next_state
				next_resources = [x for x in resources_per_event[next_state].keys()]
				if len(next_resources) > 0:
					next_state_res += "- " + " ".join([x for x in resources_per_event[next_state].keys()])

				#try:
				if next_state == last_state:
					next_action = ""
				else:
					#next_action = select_action(action_lists, q_table, next_state_res, epsilon)
					next_action = select_next_action(action_lists, q_table, next_state, next_resources, epsilon)
					timestamp = timestamp - reward
					if " - " in next_state:
						next_resource = next_state.split(" - ")[1].split(">")[0]
						resources_per_event[state.split(">")[0] + ">"][next_resource] = True
					timeline.put_nowait((timestamp, (old_state, state, next_state, start, i)))
				"""except Exception as e:
					print("Error:")
					print("i: ", i)
					print("state: ", state)
					print("action: ", action)
					print("choice: ", choice)
					print("next_state ", next_state)
					print("next_state_probability ", next_state_probability)
					print("summed_probability ", summed_probability)"""

				if (state, action) not in return_dict.keys():
					return_dict[(state, action)] = 0
				#return_dict = {k: v + reward - timestamp + start for k, v in return_dict.items()}
				return_dict = {k: v + reward for k, v in return_dict.items()}

				state = next_state
				action = next_action
				path += [state]

			# constraint positive reward
			# return_dict = add_test_pos_decl_reward(path, return_dict, declare_reward)
			#return_dict = add_sepsis_pos_decl_reward(path, return_dict, declare_reward)
			#trace = pathToTrace(path)
			#tmp, points_dict = run_all_mp_checkers_traces(trace, decl_file, txt_file, min_support_constraint, points_dict)
			#return_dict = {k: v + tmp*declare_reward for k, v in return_dict.items()}
			# update Q-Table

		for state, action in return_dict.keys():
			old_value = q_table[state][action][0]
			new_value = return_dict[(state, action)]
			# q_table[state][action][0] = ((i-1)*old_value + new_value)/i  # compute the average (learning rate alpha = 1/i)
			q_table[state][action][0] = (1-alpha)* old_value + alpha * new_value  # learning rate alpha defined above

		# print episode number
		if i % 10000 == 0:
			clear_output(wait=True)
			print(f"Episode: {i}, Q-Table: ", q_table)
			# print(path)

	print("Training finished")

	print("Exporting result")

	# MDPval_out = np.pad(MDPval, ((0,0),(0,1)), 'constant', constant_values=(0))
	MDPval = MDPval.tolist()
	for rowid, row in enumerate(MDPval):
		state = MDPval[rowid][0]
		action = MDPval[rowid][1]
		MDPval[rowid].append(q_table[state][action][0])

	header += ['q']
	MDPcsv_out = pd.DataFrame(MDPval, columns=header)
	end_time = time.time()
	total_time = end_time - start_time
	MDPcsv_out.to_csv(output_file, index = False)
	print("Result exported to: ", output_file)
	print("execution time: ", total_time)


def pathToTrace(path):
	trace = lg.Trace()
	for ev in path[1:]:
		event = lg.Event()
		if len(ev.split(" - ")) > 1:
			event["concept:name"] = ev.split(" - ")[0].rstrip().lstrip("<")
			event["org:resource"] = ev.split(" - ")[1].lstrip().rstrip(">")
		else:
			event["concept:name"] = ev.replace("<","").replace(">","")
		#event["concept:name"] = ev.replace("<","").replace(">","")
		trace.append(event)
	end = lg.Event()
	end["concept:name"] = "END"
	trace.append(end)
	return trace


def add_test_pos_decl_reward(path, return_dict, declare_reward):
	extra_reward = 0
	# unitary rewards
	if '<T>' in path:
		extra_reward += declare_reward
	if '<S>' in path:
		extra_reward += declare_reward
	# binary reward
	once = True
	for index, state in list(enumerate(path))[:-1]:
		if state == '<T>' and path[index+1] == '<S>' and once:
			extra_reward += declare_reward
			once = False
	return_dict = {k: v + extra_reward for k, v in return_dict.items()}

	return return_dict


def add_sepsis_pos_decl_reward(path, return_dict, declare_reward):
	extra_reward = 0
	once = [True, True, True, True, True]
	for index, state in list(enumerate(path))[:-1]:
		# unitary rewards
		if 'ER Registration' in state and once[0]:
			extra_reward += declare_reward
			once[0] = False
		if 'ER Triage' in state and once[1]:
			extra_reward += declare_reward
			once[1] = False
		if 'ER Sepsis Triage' in state and once[2]:
			extra_reward += declare_reward
			once[2] = False
		# binary reward
		if 'ER Registration' in state and 'ER Triage' in path[index+1] and once[3]:
			extra_reward += declare_reward
			once[3] = False
		if 'ER Triage' in state and 'ER Sepsis Triage' in path[index+1] and once[4]:
			extra_reward += declare_reward
			once[4] = False
	return_dict = {k: v + extra_reward for k, v in return_dict.items()}

	return return_dict


def select_action(action_lists, q_table, state, epsilon):
	# select action
	if random.uniform(0, 1) < epsilon:
		# exploration: action is taken randomly
		action = random.choice(action_lists[state])
	else:
		# exploitation: action is taken as the argmax
		max_val = max([v[0] for v in q_table[state].values()])
		action = random.choice([k for k, v in q_table[state].items() if v[0] == max_val])
	return action


def select_next_action(action_lists, q_table, state, resources, epsilon):
	if len(resources) > 0:
		next_action_list = list()
		q_table_dict = dict()
		for L in range(0, len(resources)+1):
			for subset in itertools.combinations(resources, L):
				s = state + "- " + " ".join(subset)
				if s in action_lists.keys():
					next_action_list.append(action_lists[s])
				if s in q_table.keys():
					for k, v in q_table[s].items():
						q_table_dict[k] = v
		next_action_list = [val for sublist in next_action_list for val in sublist]
	else:
		next_action_list = action_lists[state]
		q_table_dict = q_table[state]

	# select action
	if random.uniform(0, 1) < epsilon:
		# exploration: action is taken randomly
		action = random.choice(next_action_list)
	else:
		# exploitation: action is taken as the argmax
		max_val = max([v[0] for v in q_table_dict.values()])
		action = random.choice([k for k, v in q_table_dict.items() if v[0] == max_val])
	return action


def new_generate_q_table(MDPval):
	#extract states
	states_list = np.unique(np.transpose(MDPval)[0])
	resources_set_per_event = dict()
	resources_per_event = dict()
	# define action dictionary and q_table
	action_lists = {}
	q_table = {}
	for s in states_list:
		action_lists[s] = [] #list
		q_table[s] = {} #nested dictionary
		resources_set_per_event[s.split(">")[0] + ">"] = set()

	for rowid, row in enumerate(MDPval):
		state = MDPval[rowid,0]
		state_action = MDPval[rowid,1]
		if state_action not in action_lists[state]:
			action_lists[state].append(state_action)
			q_table[state][state_action] = [0, {}]
		next_state = MDPval[rowid, 2]
		next_state_probability = MDPval[rowid, 3]
		next_state_reward = MDPval[rowid, 4]
		q_table[state][state_action][1][next_state] = [next_state_probability, next_state_reward]
		if " - " in next_state:
			next_res = next_state.split(" - ")[1].replace(">", "").strip()
			resources_set_per_event[state.split(">")[0] + ">"].add(next_res)

	for state, lis in resources_set_per_event.items():
		resources_per_event[state.split(">")[0] + ">"] = dict()
		lis = list(lis)
		lis.sort()
		for res in lis:
			resources_per_event[state.split(">")[0] + ">"][res] = False

	resources_per_event["<END>"] = {}

	return states_list, action_lists, q_table, resources_per_event


# def update_flag_activity(flag_activity_dict):

def compute_constraint_reward(state, next_state, reward, flag_activity_dict, unitary_decl_reward_dict, binary_decl_reward_dict):
	current_activity = get_activity(state)
	next_activity = get_activity(next_state)
	# unitary constraint
	if next_activity in flag_activity_dict.keys():
		old_flag = flag_activity_dict[next_activity]
		flag_activity_dict[next_activity] += 1
		flag = flag_activity_dict[next_activity]
		lower_bound, upper_bound, pos_reward, neg_reward = unitary_decl_reward_dict[next_activity]
		if old_flag <= lower_bound < flag:
			reward += pos_reward
		if flag >= upper_bound:
			reward += neg_reward
	# binary constraints
	if current_activity in binary_decl_reward_dict.keys():
		constrained_next_activity, pos_reward, neg_reward = binary_decl_reward_dict[current_activity]
		if next_activity == constrained_next_activity:
			reward += pos_reward
		else:
			reward += neg_reward
	check = 0

	return reward, flag_activity_dict


def get_activity(state):
	activity_resource = state.replace('<','').replace('>','')
	activity_resource_list = activity_resource.split('-')
	activity = activity_resource_list[0]

	return activity


def generate_q_table(MDPval):
	#extract states
	states_list = np.unique(np.transpose(MDPval)[0])

	# define action dictionary and q_table
	action_lists = {}
	q_table = {}
	for s in states_list:
		action_lists[s] = [] #list
		q_table[s] = {} #nested dictionary

	for rowid, row in enumerate(MDPval):
		state = MDPval[rowid,0]
		state_action = MDPval[rowid,1]
		if state_action not in action_lists[state]:
			action_lists[state].append(state_action)
			q_table[state][state_action] = 0

	return states_list, action_lists, q_table

# def get_rewards(MDPval): #TO REMOVE
#         # target states
#         new_states_list = np.unique(np.transpose(MDPval)[2])
#
#         # define reward dictionary
#         reward_list = {}
#         for rowid, row in enumerate(MDPval):
#             if
#             state = MDPval[rowid, 0]
#             next_state = MDPval[rowid, 2]
#             state_reward = MDPval[rowid, 4]
#             reward_list[state][next_state] = state_reward
#
#         return reward_list


if __name__ == "__main__":
	main()




