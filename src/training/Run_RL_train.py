import random

import numpy as np
import pandas as pd
import math
from IPython.display import clear_output

def main():

    print("Importing the model")

    #import csv, must have these columns (s,a,s',p,r)
    input_file = '../input_data/simple_model_MDP_r3.csv'
    output_file = '../output_data/simple_model_MDP_r3_Q.csv'

    MDPcsv = pd.read_csv(input_file)

    #convert into array
    MDPval = MDPcsv.values
    header = list(MDPcsv.columns.array)

    #Q-matrix and co.
    states_list, action_lists, q_table = generate_q_table(MDPval)

    #rewards #TO REMOVE
    #reward_list = get_rewards(MDPval)

    #RL hyperparameters
    alpha = 0.1 # learning rate
    gamma = 0.7 # discount rate
    epsilon = 0.1 # discovery rate

    print("Training the agent")

    #loop on episodes
    for i in range(1,1000001):
        state = 'Start' #initial state
        reward = 0 #initial reward
        done = False

        while not done:
            if random.uniform(0,1) < epsilon:
                #exploration: action is taken randomly
                action = random.choice(action_lists[state])
            else:
                #exploitation: action is taken as the argmax
                max_val = max(q_table[state].values())
                action = random.choice([k for k, v in q_table[state].items() if v == max_val])

            # these three variables manage stochastic decision
            summed_probability = 0
            next_state_probability = 0
            choice = random.uniform(0,1)
            next_state = 'Undefined'
            # stochastic decision
            for rowid, row in enumerate(MDPval):
                if MDPval[rowid,0] == state and MDPval[rowid,1] == action:
                    next_state_probability = MDPval[rowid,3]
                    if summed_probability < choice and choice <= summed_probability + next_state_probability:
                        next_state = MDPval[rowid,2]
                        reward = MDPval[rowid,4]
                    summed_probability = summed_probability + next_state_probability

            try:
                if next_state == 'Stop':
                    done = True
                    next_max = 0
                else:
                    next_max = max(q_table[next_state].values())
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
            old_value = q_table[state][action]
            # one may use alpha/(math.log(1,10)+1)) instead!?
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state][action] = new_value
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
        MDPval[rowid].append(q_table[state][action])

    header += ['q']
    MDPcsv_out = pd.DataFrame(MDPval, columns=header)

    MDPcsv_out.to_csv(output_file, index = False)
    print("Result exported to: ", output_file)

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




