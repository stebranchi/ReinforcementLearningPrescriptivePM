import os
# import re
import pandas as pd
import numpy as np

state_label = 's'
action_label = 'a'
next_state_label = 's\''
reward_label = 'reward'
start_state = '<>'
end_state_particle = 'END'
loop_state = 'loop'
dead_end_state = 'missing'
end_state = 'end'

only_events = False
max_length_path = 18

def main():
	# define input output file
	csv_q_value = os.path.join("..", "..", "cluster_data", "output_policies", "BPI2012_50_ordered_linear_scale_factor_bug_fixed.csv")
	csv_policy = csv_q_value.replace('.csv', '_policy_only_events.csv' if only_events else '_policy.csv')
	csv_output = csv_policy.replace(".csv", '_paths.csv')
	# import MDP df
	df = pd.read_csv(csv_q_value, sep=',')

	if only_events:
		# modify take state only events
		df = only_event_MDP(df)

	if ("policy" not in df.keys()) or only_events:
		# compute policy values 1/0 from q value
		df = compute_policy(df)
		df.to_csv(csv_policy, sep=',')


	# filter df on policy = 1
	df_on_policy = filter_on_policy(df)

	# extract transitions from df into a dict
	next_state_dict, end_state_reward = extract_next_state_dict(df)

	# contruct a matrix, every row is a possible path
	path_matrix = build_path_matrix(next_state_dict, end_state_reward)
	# print(df.head(3).to_numpy())

	# export result to csv
	write_file(path_matrix, csv_output)


def print_statistics(matrix):
	running_state = "running"
	stats_dict = {end_state: 0, loop_state: 0, dead_end_state: 0, running_state: 0}
	for row in matrix:
		if row[-1] == end_state:
			stats_dict[end_state] += 1
		elif row[-1] == loop_state:
			stats_dict[loop_state] += 1
		elif row[-1] == dead_end_state:
			stats_dict[dead_end_state] += 1
		else:
			stats_dict[running_state] += 1
	print("Matrix length:", len(matrix), ", stats:", stats_dict)

def filter_on_policy(df):
	df = df.loc[df["policy"] == 1]
	return df

def only_event_MDP(df):
	df[state_label] = only_event_column(df[state_label])
	df[next_state_label] = only_event_column(df[next_state_label])
	return df

def only_event_column(state_column):
	new_state_column = [only_event_state(s) for s in state_column]
	return new_state_column

def only_event_state(state):
	sep = " | "
	if sep in state:
		state = state.split(sep)[0] + ">"
	return state

def compute_policy(df):
	state_set = set(df[state_label])
	max_q_dict = {}
	for s in state_set:
		filtered_df = df[[state_label, "q"]].loc[df[state_label] == s]
		max_q_dict[s] = max(filtered_df["q"])
	df['max_q'] = df['s'].map(max_q_dict)
	df['policy'] = np.where(df['q'] == df['max_q'], 1, 0)
	df = df.drop(columns=['max_q'])
	return df

def convert_to_np(matrix):
	np_matrix = np.asarray([np.asarray(row) for row in matrix])
	return np_matrix

def write_file(matrix, csv):
	with open(csv, 'w') as f:
		for row in matrix:
			line_output = ','.join(row) + "\n"
			f.write(line_output)

def build_path_matrix(next_state_dict, end_state_reward):
	done = False
	matrix = [[start_state]]
	i = 0
	while (not done) and i < max_length_path:
		print("i:", i)
		# compute statistics on the path matrix
		print_statistics(matrix)
		new_matrix = []
		for row in matrix:
			last_state = row[-1]
			if check_complete_state(last_state):
				new_matrix.append(row)
			elif check_end_state(last_state):
				new_matrix.append(row + [end_state])
			elif last_state in next_state_dict.keys():
				for next_state in next_state_dict[last_state]:
					next_state_to_check = next_state[-1] if type(next_state) is list else next_state
					if end_state_particle in next_state_to_check:
						next_state_to_add = [next_state[0], next_state[1] + " " + str(end_state_reward[next_state_to_check][0])]
					else:
						next_state_to_add = next_state
					if next_state in row:
						new_matrix.append(row + next_state_to_add + [loop_state])
					else:
						new_matrix.append(row + next_state_to_add)
			else:
				new_matrix.append(row + [dead_end_state])
		matrix = new_matrix
		i += 1
		done = check_complete_matrix(matrix)
	print("i:", i)
	# compute statistics on the path matrix
	print_statistics(matrix)

	return matrix

def check_complete_matrix(matrix):
	end_condition_list = [check_complete_state(row[-1]) for row in matrix]
	outcome = all(end_condition_list)
	return outcome

def check_end_state(state):
	outcome = end_state_particle in state
	return outcome

def check_complete_state(state):
	outcome = state == loop_state or \
			  state == dead_end_state or \
			  state == end_state
	return outcome

def extract_next_state_dict(df):
	state_set = set(df[state_label])
	end_state_dict = df.loc[(df['policy'] == 1) & (df[next_state_label].str.contains('END')), [next_state_label, reward_label]].set_index(next_state_label).T.to_dict('list')
	next_state_dict = {}
	for s in state_set:
		# filtered_df = df[[state_label, next_state_label]].loc[df[state_label] == s]
		filtered_df = df.loc[(df[state_label] == s) & (df['policy'] == 1), [state_label, action_label, next_state_label]]
		next_state_list = [[a,b] for a,b in zip(filtered_df[action_label], filtered_df[next_state_label])]
		next_state_dict[s] = next_state_list
	return next_state_dict, end_state_dict


if __name__ == "__main__":
	main()