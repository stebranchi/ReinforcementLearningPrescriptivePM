import random
import os

import numpy as np
import pandas as pd
import time
import math
from IPython.display import clear_output
from pm4py.objects.log import obj as lg

def main():
    # method: Q-learning or SARSA
    method = 'Q-learning'
    #method = 'SARSA'
    print("Method:", method)
    start_time = time.time()
    print("Importing the model")

    #import csv, must have these columns (s,a,s',p,r) these are respectively: state, action, next state, probability, reward(e.g. -time execution)
    # input_file = os.path.join("..", "input_data", "test", "model", "test_model_1_r.csv")
    # output_file = os.path.join("..", "output_data", "test", "Q_values", "test_model_1_r_Q_SARSA_decl_pos.csv")

    #input_file = os.path.join("..", "input_data", "sepsis", "model", "sepsis_model_MDP_r.csv")
    #output_file = os.path.join("..", "output_data", "sepsis", "Q_values", "sepsis_model_MDP_r_decl2_Q_bis.csv")

    #input_file = os.path.join("..", "data", "mdp", "sepsis_model_MDP_r.csv")
    #output_file = os.path.join("..", "data", "output", "sepsis_model_MDP_r_decl2_Q.csv")

    #input_file = os.path.join("..", "final_evaluation", "BPI", "Mdp", "Trimmed BPI_2012 mdp training 60 r005.csv")
    #output_file = os.path.join("..", "final_evaluation", "BPI", "Output", "Trimmed BPI_2012 mdp training 60 r005 Qlearn.csv")

    input_file = "Trimmed BPI_2012 mdp training 60 r005 var90 scaled3.csv"
    output_file = "Trimmed BPI_2012 mdp training 60 r005 var90 scaled3 policy.csv"

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
    states_list, action_lists, q_table = new_generate_q_table(MDPval)

    # constraint rewards
    declare_reward = 10000
    min_support_constraint = 70

    #rewards #TO REMOVE
    #reward_list = get_rewards(MDPval)

    #RL hyperparameters
    alpha_max = 0.1 # initial learning rate
    alpha_min = 0.001 # final learning rate
    gamma = 1 # discount rate, discount near to 1 avoids loops
    epsilon = 0.1 # discovery rate

    print("Training the agent")

    #loop on episodes
    number_of_runs = 10000000
    points_dict = None
    for i in range(1, number_of_runs):
        state = first_state # initial state
        action = select_action(action_lists, q_table, state, epsilon) # initial action
        path = [state]

        reward = 0  # initial reward
        done = False
        # linearly variable alpha: alpha=alpha_max when i=1, alpha=alpha_min when i=number_of_runs
        alpha = alpha_max - (alpha_max - alpha_min)*(i-1)/(number_of_runs-1)

        while not done:
            # these three variables manage stochastic decision
            summed_probability = 0
            next_state_probability = 0
            choice = random.uniform(0, 1)
            next_state = 'Undefined'
            # stochastic decision
            list_of_possibilities = q_table[state][action][1]
            list_of_possibilities = q_table[state][action][1]
            p = np.array([v[0] for x,v in list_of_possibilities.items()])
            p /= p.sum()
            next_state = np.random.choice([x for x in list_of_possibilities.keys()], p=p)
            reward = list_of_possibilities[next_state][1]

            try:
                if next_state == last_state:
                    done = True
                    next_max = 0 # for Q-learning
                    next_value = 0 # for SARSA
                    next_action = ""
                else:
                    next_max = max([v[0] for v in q_table[next_state].values()]) # for Q-learning
                    # select next action
                    next_action = select_action(action_lists, q_table, next_state, epsilon)
                    next_value = q_table[next_state][next_action][0] # for SARSA
            except Exception as e:
                print("Error:")
                print("i: ", i)
                print("state: ", state)
                print("action: ", action)
                print("choice: ", choice)
                print("next_state ", next_state)
                print("next_state_probability ", next_state_probability)
                print("summed_probability ", summed_probability)

            # update Q-Table
            old_value = q_table[state][action][0]
            if method == 'Q-learning':
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)  # TD Q-Learning
            elif method == 'SARSA':
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_value)  # TD SARSA
            q_table[state][action][0] = new_value

            if next_state not in q_table.keys():
                # this is the case when the state is not present in the policy model
                reward = -5
                done = True
            """elif len(q_table[next_state]) == 0:
                # this is the case when the state has no possible action in the intersection of reward and policy models
                reward = -500
                done = True"""

            if next_state in path:
                reward = -5
                done = True

            state = next_state
            action = next_action
            path.append(state)

        # print episode number
        if i % 10000 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}, Q-Table: ", q_table)

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
        if len(ev.split("-")) > 1:
            event["concept:name"] = ev.split("-")[0].rstrip().lstrip("<")
            event["org:resource"] = ev.split("-")[1].lstrip().rstrip(">")
        else:
            event["concept:name"] = ev.replace("<", "").replace(">", "")
        trace.append(event)
    end = lg.Event()
    end["concept:name"] = "END"
    trace.append(end)
    return trace


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


def new_generate_q_table(MDPval):
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
            q_table[state][state_action] = [0, {}]
        next_state = MDPval[rowid, 2]
        next_state_probability = MDPval[rowid, 3]
        next_state_reward = MDPval[rowid, 4]
        q_table[state][state_action][1][next_state] = [next_state_probability, next_state_reward]

    return states_list, action_lists, q_table


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


if __name__== "__main__":
  main()




