import random
import os

import numpy as np
import pandas as pd
import math
from pm4py.objects.log import log as lg
from IPython.display import clear_output
from src.mp_checkers.test_mp_checkers_traces import run_all_mp_checkers_traces

def main():

    print("Importing the model")

    #import csv, must have these columns (s,a,s',p,r) these are respectively: state, action, next state, probability, reward(e.g. -time execution)
    input_file = os.path.join("..", "data", "mdp", "sepsis_model_MDP_r_Q.csv")
    decl_file = os.path.join("..", "data", "declare", "Sepsis Cases - Event Log_training_resources_w_ends_0_1_binary+.decl")
    txt_file = os.path.join("..", "data", "declare", "Sepsis Cases - Event Log_training_resources_w_ends_0_1_binary+.txt")
    output_file = os.path.join("..", "data", "output", "sepsis_model_MDP_r_decl2_Q.csv")

    # input_file = '../input_data/new_SB/SB_simple_model_MDP_r1.csv'
    # output_file = '../output_data/new_SB/SB_simple_model_MDP_p1_r1_Q.csv'

    # first and last state definition
    # first_state = 'Start' #  first state with manual simple model
    first_state = '<>'      #  first state with new_SB simple model
    # last_state = 'Stop'   #  last state with manual simple model
    last_state = '<END>'    #  last state with new_SB simple model

    # activity constraint rewards
    ## STEFANO: qui gestisco i reward, sono positivi/negativi per constraint positive/negative, a noi servirà gestire solo quelli positivi
    unitary_reward = 10000
    constraint_dict = None
    binary_reward = unitary_reward
    reward_last_event = unitary_reward # special constraint for the reaching of END, avoids loops
    unitary_decl_reward_dict = {}
    unitary_decl_reward_dict['ER Registration'] = [0, 2, unitary_reward, -unitary_reward] # lower bound, upper bound, positive reward, negative reward
    unitary_decl_reward_dict['ER Sepsis Triage'] = [0, 2, unitary_reward, -unitary_reward]
    unitary_decl_reward_dict['ER Triage'] = [0, 3, unitary_reward, -unitary_reward]
    binary_decl_reward_dict = {}
    binary_decl_reward_dict['ER Registration'] = ['ER Triage', binary_reward, -binary_reward]
    binary_decl_reward_dict['ER Triage'] = ['ER Sepsis Triage', binary_reward, -binary_reward]

    # load the policy model
    MDPcsv = pd.read_csv(input_file)

    #convert into array
    MDPval = MDPcsv.values
    header = list(MDPcsv.columns.array)

    #Q-matrix and co.
    states_list, action_lists, q_table = new_generate_q_table(MDPval)

    #rewards #TO REMOVE
    #reward_list = get_rewards(MDPval)

    #RL hyperparameters
    alpha_max = 0.1 # initial learning rate
    alpha_min = 0.1 # final learning rate
    gamma = 0.9 # discount rate, discount near to 1 avoids loops
    epsilon = 0.5 # discovery rate

    print("Training the agent")

    #loop on episodes
    number_of_runs = 100000
    for i in range(1, number_of_runs):
        trace = lg.Trace()
        state = first_state
        reward = 0  # initial reward
        done = False
        if i % 100 == 0:
            print(i)
        # this is a flag used to keep tracks of constraint
        flag_activity_dict = {'ER Registration': 0, 'ER Triage': 0, 'ER Sepsis Triage': 0}
        # linearly variable alpha: alpha=alpha_max when i=1, alpha=alpha_min when i=number_of_runs
        alpha = alpha_max - (alpha_max - alpha_min)*(i-1)/(number_of_runs-1)

        while not done:
            if random.uniform(0,1) < epsilon:
                #exploration: action is taken randomly
                action = random.choice(action_lists[state])
            else:
                #exploitation: action is taken as the argmax
                max_val = max([v[0] for v in q_table[state].values()])
                action = random.choice([k for k, v in q_table[state].items() if v[0] == max_val])

            # these three variables manage stochastic decision
            summed_probability = 0
            next_state_probability = 0
            choice = random.uniform(0, 1)
            next_state = 'Undefined'
            # stochastic decision
            list_of_possibilities = q_table[state][action][1]
            for possible_next_state in list_of_possibilities.keys():
                next_state_probability = list_of_possibilities[possible_next_state][0]
                if summed_probability < choice and choice <= summed_probability + next_state_probability:
                    next_state = possible_next_state
                    reward = list_of_possibilities[next_state][1]
                    # computation of special reward based declarative constraints
                    ## STEFANO: qui calcolo il reward dovuto alle constraint soddisfatte ad ogni step
                    #reward, flag_activity_dict = compute_constraint_reward(state, next_state, reward,
                    #                                                       flag_activity_dict, unitary_decl_reward_dict,
                    #                                                       binary_decl_reward_dict)
                    event = lg.Event()
                    event["concept:name"] = next_state.split("-")[0].replace("<", "")
                    if len(next_state.split("-")) > 1:
                        event["org:resource"] = next_state.split("-")[1].replace(">", "")
                    trace.append(event)
                    reward, constraint_dict = compute_reward_checker(trace, reward, txt_file, decl_file, constraint_dict)
                summed_probability = summed_probability + next_state_probability


            try:
                if next_state == last_state:
                    done = True
                    next_max = 0
                    reward += reward_last_event # this special END reward helps in going straight to the END avoiding reward-seeking  detours
                else:
                    next_max = max([v[0] for v in q_table[next_state].values()])
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
            # reward = reward_list[next_state] #TO REMOVE
            old_value = q_table[state][action][0]
            # one may use alpha/(math.log(1,10)+1)) instead!?
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)  # this is temporal-difference formula
            q_table[state][action][0] = new_value
            state = next_state

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

    MDPcsv_out.to_csv(output_file, index = False)
    print("Result exported to: ", output_file)


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


def compute_reward_checker(trace, reward, txt_path, decl_path):
    tmp_r = run_all_mp_checkers_traces(trace, decl_path, txt_path)

    return reward + tmp_r


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




