import random
import os

import numpy as np
import pandas as pd
import math
from IPython.display import clear_output

def main():

    #import csv, must have these columns (s,a,s',p,r)
    reward_file = os.path.join("..", "data", "output", "new_tests", "BPI_2012_log_eng_preprocessed_filtered_old.csv")
    policy_file = os.path.join("..", "data", "output", "new_tests", "BPI_2012_log_eng_preprocessed_filtered_old.csv")
    # reward_file = '../input_data/new_SB/SB_simple_model_MDP_r1.csv'
    # policy_file = '../output_data/new_SB/SB_simple_model_MDP_p9_r3_Q.csv'
    random_policy = False

    # first and last state definition
    first_state_list = ['<>']     #  first state with new_SB simple model
    # first_state_list = ['<ER Registration-A>', '<ER Registration-L>']
    last_state = '<END>'    #  last state with new_SB simple model # was 'Stop'

    print("Importing the model")

    MDP_reward = pd.read_csv(reward_file)
    MDP_policy = pd.read_csv(policy_file)

    #convert into array
    MDP_reward_val = MDP_reward.values
    MDP_policy_val = MDP_policy.values

    #Q-matrix and co.
    states_list, action_dict, reward_dict = get_rules(MDP_policy_val, MDP_reward_val, random_policy)

    #rewards #TO REMOVE
    #reward_list = get_rewards(MDP_reward_val)

    if random_policy:
        print("Testing random policy")
    else:
        print("Testing input policy:", policy_file)

    runs = 0 # variables to count effective runs, if reward and policy are different some runs may be rejected because arrives in a state with no action available
    total_reward = 0
    number_of_runs = 100001
    set_of_paths = []
    #loop on episodes
    for i in range(1, number_of_runs):
        single_run_reward = 0
        state = random.choice(first_state_list)
        reward = 0 #initial reward
        done = False
        wrong_path = False # path which has not defined reward
        path = [state]

        # from here start one trace
        while not done:
            if state not in action_dict.keys():
                # this is the case when the state is not present in the policy model
                break
            elif len(action_dict[state]) == 0:
                # this is the case when the state has no possible action in the intersection of reward and policy models
                break

            # action selection
            if random_policy:
                #exploration: action is taken randomly
                action, max_val = random.choice(action_dict[state])
            else:
                max_val = max([value for action, value in action_dict[state]])
                action = random.choice([k for k, v in action_dict[state] if v == max_val])

                # max_val = max([v[0] for v in q_table[state].values()])
                # action = random.choice([k for k, v in q_table[state].items() if v[0] == max_val])

            # these three variables manage stochastic decision
            summed_probability = 0
            next_state_probability = 0
            choice = random.uniform(0, 1)
            next_state = 'Undefined'
            # stochastic decision
            list_of_possibilities = reward_dict[state][action]
            for possible_next_state in list_of_possibilities.keys():
                next_state_probability = list_of_possibilities[possible_next_state][0]
                if summed_probability < choice and choice <= summed_probability + next_state_probability:
                    next_state = possible_next_state
                    reward = list_of_possibilities[next_state][1]
                summed_probability = summed_probability + next_state_probability

            try:
                # if next_state == last_state: # old condition
                if last_state in next_state:  # new condition, it is compatible with state = full prefix
                    done = True

            except Exception as e:
                print("Error:")
                print("i: ", i)
                print("run:", runs)
                print("state: ", state)
                print("action: ", action)
                print("choice: ", choice)
                print("next_state ", next_state)
                print("next_state_probability ", next_state_probability)
                print("summed_probability ", summed_probability)

            # update total_reward
            # reward = reward_list[next_state]#TO REMOVE
            single_run_reward -= reward
            state = next_state
            path += [state]

        if done:
            runs += 1
            total_reward += single_run_reward
            if path not in set_of_paths:
                set_of_paths.append(path)

        # print episode number
        if i % 10000 == 0:
            clear_output(wait=True)
            if runs > 0:
                avg_reward = total_reward/runs
            else:
                avg_reward = 0
            print(f"Episode: {i}, run:", runs, ", Total reward: ", total_reward, ", average reward: ", avg_reward)
            print("Last path: ", path)

    print("Testing finished")
    if not random_policy:
        print("Paths: ")
        for path in set_of_paths:
            print(path)


def get_rules(MDP_policy_val, MDP_reward_val, random_policy):
    #extract states
    policy_states_list = np.unique(np.transpose(MDP_policy_val)[0])
    reward_states_list = np.unique(np.transpose(MDP_reward_val)[0])

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

    return states_list, action_dict, reward_dict



if __name__== "__main__":
  main()




