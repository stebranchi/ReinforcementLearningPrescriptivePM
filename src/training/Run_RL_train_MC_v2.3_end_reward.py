import random
import os

import numpy as np
import pandas as pd
import time
import math
from IPython.display import clear_output
from pm4py.objects.log import obj as lg
from sklearn.preprocessing import MinMaxScaler


# global variables def: header label of the MDP csv
state_label = "s"
action_label = "a"
next_state_label = "s\'"
probability_label = "p_r"
reward_label = "reward"
n_occurences_label = "number_occurrences"
q_value_label = "q"
scale_factor_label = "scale_factor"
policy_label = "policy"

# global variables def: first and last state definition
first_state = '<>'  # first state with new_SB simple model
last_state_particle = 'END'  # last state with new_SB simple model

# global variables def: parameter to discourage to long trace simulations or loops
max_trace_length = 100
exceeding_traces_length_penalty = -1000
loop_penalty = 0

# global variables def: RL hyperparameters
alpha_max = 0.1  # initial learning rate
alpha_min = 0.001  # final learning rate
gamma = 0.9  # discount rate, discount near to 1 avoids loops
epsilon = 0.1  # discovery rate
number_of_runs = 10000000  # number of episodes simulated during training
scale_factor_type = "linear"  # types of scale factor: none, linear, exp
scale_factor_exp_weight = 3  # weight for the exp scale factor
normalize_reward = False  # use minmaxscaler on reward?
change_zero_reward = False  # if minmaxscaler on reward is used, apply also to zero reward?

def main():
    print("Importing the model")
    start_time = time.time()

    input_file = os.path.join("..", "..", "cluster_data", "output_mdps",
                              "BPI2012_log_eng_positional_cumulative_squashed_training_80_preprocessed_scaled.csv")
    output_file = os.path.join("..", "..", "cluster_data", "output_policies",
                               "BPI2012_log_eng_positional_cumulative_none_scale_factor.csv")
    # load the policy model
    MDP_df = pd.read_csv(input_file)

    # add columns to df: q-value, scale_factor, and normalize_reward
    MDP_df = preprocess_df(MDP_df)

    #Q-matrix and co.
    # states_list, state_action_dict, MDP_df = generate_q_df(MDP_df)
    states_list, state_action_dict, q_table = generate_q_table(MDP_df)

    print("Training the agent")

    #loop on episodes
    for i in range(1, number_of_runs):
        state = first_state # initial state
        action = select_action(state_action_dict, q_table, state, epsilon) # initial action
        return_dict = {}
        path = [state]

        reward = 0  # initial reward
        done = False
        # inversely proportional alpha: mean compute the average return
        # alpha = 1 / i
        # linearly variable alpha: alpha=alpha_max when i=1, alpha=alpha_min when i=number_of_runs
        alpha = alpha_max - (alpha_max - alpha_min)*(i-1)/(number_of_runs-1)
        trace_i = 0
        state_action_first_appear_list = []
        while not done:
            # these three variables manage stochastic decision
            summed_probability = 0
            next_state_probability = 0
            choice = random.uniform(0, 1)
            next_state = ''
            # stochastic decision
            list_of_possibilities = q_table[state][action]["next_state_dict"]
            # list of probabilities transition for the list of possibilities (v is a dict {"p": p, "r": r,...})
            p = np.array([v[probability_label] for x, v in list_of_possibilities.items()])
            p /= p.sum()  # renormalization to avoid machine missing digits
            # stochastically choose next_state
            next_state = np.random.choice([x for x in list_of_possibilities.keys()], p=p)
            reward = list_of_possibilities[next_state][reward_label]

            try:
                if last_state_particle in next_state:
                    done = True
                    next_action = ""
                elif next_state not in states_list:
                    # this is the case when the state is not present in the policy model
                    reward = 0
                    done = True
                #elif len(q_table[next_state]) == 0:
                #    # this is the case when the state has no possible action in the intersection of reward and policy models
                #    reward = -500
                #    done = True
                else:
                    next_action = select_action(state_action_dict, q_table, next_state, epsilon)
            except Exception as e:  # for degug
                print("Error:")
                print("i: ", i)
                print("state: ", state)
                print("action: ", action)
                print("choice: ", choice)
                print("next_state ", next_state)
                print("next_state_probability ", next_state_probability)
                print("summed_probability ", summed_probability)

            # Check if is going through a loop
            if next_state in path:
                reward = loop_penalty
                done = True

            # Check if trace is too long
            if trace_i > max_trace_length:
                reward = exceeding_traces_length_penalty
                done = True

            # add to list of returns
            if (state, action) not in return_dict.keys():
                return_dict[(state, action)] = 0
                state_action_first_appear_list.append((state, action))
            return_dict = {k: return_dict[k] + (reward * gamma ** (len(state_action_first_appear_list) - (n+1))) for n, k in enumerate(state_action_first_appear_list)}

            state = next_state
            action = next_action
            path += [state]
            trace_i += 1

        # update Q-Table
        for state, action in return_dict.keys():
            q_value = q_table[state][action]["q"]
            scale_factor = q_table[state][action][scale_factor_label]
            # scaled factor takes into accunt the confidence of the transition (number of occurences in the log)
            q_value_update = return_dict[(state, action)] * scale_factor
            q_table[state][action]["q"] = (1-alpha) * q_value + alpha * q_value_update  # update q-value
            # learning rate alpha is defined above

        # print episode number
        if i % 10000 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")
            # print(path)

    print("Training finished")

    print("Exporting result")
    # update MDP_df
    for s, a_list in state_action_dict.items():
        for a in a_list:
            q_value = q_table[s][a]["q"]
            MDP_df.loc[(MDP_df[state_label] == s) & (MDP_df[action_label] == a), [q_value_label]] = q_value
    # compute policy
    MDP_df = compute_policy(MDP_df)
    # export to csv
    MDP_df.to_csv(output_file, index=False)
    print("Result exported to: ", output_file)

    end_time = time.time()
    total_time = end_time - start_time
    print("execution time: ", total_time)

# function to take the data as group and perform aggregation
def total_occurrences(group):
    g = group[n_occurences_label].agg('sum')
    group['sum_n_occurrences'] = g
    return group

def preprocess_df(df):
    # add q-value with zeros
    df[q_value_label] = 0

    # compute total number of occurrences per action
    df = df.groupby([state_label, action_label]).apply(total_occurrences)

    # define scale factor
    # redefine scale factor label for the df to include the type of scaling
    # scale_factor_label = scale_factor_label + "_" + scale_factor_type
    if scale_factor_type == "none":
        df[scale_factor_label] = 1
    elif scale_factor_type == "linear":
        minmax_scale = MinMaxScaler(feature_range=(0, 1))
        df[scale_factor_label] = minmax_scale.fit_transform(df[['sum_n_occurrences']])
    elif scale_factor_type == "exp":
        w = scale_factor_exp_weight
        # scale_factor_label += "_" + str(w)
        df[scale_factor_label] = df["sum_n_occurrences"].apply(lambda x: -2 * (math.exp(-x/w)/(1+math.exp(-x/w))) + 1)
    elif scale_factor_type == "step":
        df[scale_factor_label] = df["sum_n_occurrences"].apply(lambda x: 0 if x <= 3 else 1)

    # normalize reward column
    if normalize_reward:
        minmax_reward = MinMaxScaler(feature_range=(0, 1))
        # df[reward_label] = np.where(df[reward_label] == 0, 0, minmax_scale.fit_transform(df[[reward_label]]))
        df['scaled_reward'] = minmax_reward.fit_transform(df[[reward_label]])
        if change_zero_reward:
            df['new_reward'] = df['scaled_reward']
        else:
            df['new_reward'] = np.where(df[reward_label] == 0, 0, df['scaled_reward'])
        df[reward_label] = df['new_reward']  # for debug
        df = df.drop(columns=['scaled_reward', 'new_reward'])
    df = df.drop(columns=['sum_n_occurrences'])
    return df

def generate_q_table(MDP_df):
    """
    q-table is a dict (state) of dict (action)
    each values is a couple: the first value is the q-value, the second value is a dict
    the dict is {next_state: {"p": p, "r": r}}
    in total {s: {a: {"q": 0, "next_state_dict": {s': {"p": p, "r": r}}, "scale_factor": scale_factor}}}
    """

    #extract states
    states_list = np.unique(MDP_df[state_label])
    # define action dictionary and q_table
    q_table = {}
    state_action_dict = {}
    for s in states_list:
        state_action_dict[s] = np.unique(MDP_df.loc[MDP_df[state_label] == s, [action_label]])
        q_table[s] = {}

    # build q_table
    for s, a_list in state_action_dict.items():
        for a in a_list:
            next_state_df = MDP_df.loc[(MDP_df[state_label] == s) & (MDP_df[action_label] == a),
                                       [next_state_label, probability_label, reward_label, scale_factor_label]]
            scale_factor = next_state_df[scale_factor_label].to_numpy()[0]
            next_state_df = next_state_df.drop(columns=[scale_factor_label])
            next_state_dict = create_next_state_dict(next_state_df)
            q_table[s][a] = {"q": 0, "next_state_dict": next_state_dict, scale_factor_label: scale_factor}

    return states_list, state_action_dict, q_table

def create_next_state_dict(next_state_df):
    # create dict {next_state: {"p": p, "r": r, "n_occ": n_occ}}
    next_state_df.set_index(next_state_label, inplace=True)
    next_state_dict = next_state_df.to_dict('index')
    return next_state_dict

def select_action(state_action_dict, q_table, state, epsilon):
    # select action
    if random.uniform(0, 1) < epsilon:
        # exploration: action is taken randomly
        action = random.choice(state_action_dict[state])
    else:
        # exploitation: action is taken as the argmax
        # take the max q-value for that state (q_table[state] is a dict {a: {"q": q, ...}}
        max_q_value = max([v["q"] for v in q_table[state].values()])
        # select randomly on of the best action
        action = random.choice([k for k, v in q_table[state].items() if v["q"] == max_q_value])
    return action

def compute_policy(df):
	state_set = set(df[state_label])
	max_q_dict = {}
	for s in state_set:
		filtered_df = df.loc[df[state_label] == s,[state_label, q_value_label]]
		max_q_dict[s] = max(filtered_df[q_value_label])
	df['max_q'] = df[state_label].map(max_q_dict)
	df[policy_label] = np.where(df[q_value_label] == df['max_q'], 1, 0)
	df = df.drop(columns=['max_q'])
	return df

if __name__== "__main__":
    main()




