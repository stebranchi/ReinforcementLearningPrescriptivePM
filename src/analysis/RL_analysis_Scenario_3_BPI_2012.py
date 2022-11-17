import os
import pandas as pd
import numpy as np
from datetime import datetime

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

# TODO:Cambiare i vari attributi, non ci serve più la durata visto che ho già il reward, gli attributi non servono più ma solo il cluster

case_id_label = 'concept:name'
event_label = 'concept:name'
lifecycle_label = 'lifecycle:transition'
timestamp_label = 'time:timestamp'
attributes_label_list = ['amount', 'n_calls_after_offer', 'n_calls_missing_doc', 'number_of_offers', 'number_of_sent_back', 'W_Fix_incoplete_submission']
amount_label = 'amount'
duration_label = 'duration'
cost_factor = 0.005
rounding = 1

def main():
    # import log
    # log_path = os.path.join("..", "input_data", "BPM_Scenarios", "Scenario_3-BPI_2012", "v4", "log", "BPI_2012_log_eng_training_80_mid_preprocessed_number_incr_wfix.xes")
    log_path = os.path.join("..", "input_data", "BPM_Scenarios", "Scenario_3-BPI_2012", "v4", "log", "BPI_2012_log_eng_testing_20_mid_preprocessed_number_incr_wfix.xes")
    log = xes_importer.apply(log_path)

    all_traces_analysis = False  # global analysis of all traces
    only_events = False  # means no resources are used in definition of states


    # import policy csv, must have these columns (s,a,s',p,r,q)
    policy_file = os.path.join("..", "output_data", "BPM_Scenarios", "Scenario_3-BPI_2012", "v4", "Trimmed BPI_2012 mdp training 80 n_wfix scaled3 policy.csv")

    MDP_policy = pd.read_csv(policy_file)
    MDP_policy_val = MDP_policy.values  # converts it into array
    policy_rules_dict = get_policy_rules(MDP_policy_val, only_events)  # policy rules give a list of possible next state for each current state

    events_and_attributes = [('START', [])] # <START>
    # events_and_attributes = [('O_CREATED', ['medium', 0, 0, 0, 0])] # <O_CREATED - medium - 0 - 0 - 0>
    initial_states = [event_attributes_to_state(event, attributes, only_events) for event, attributes in events_and_attributes]
    analysis_sc3(log, policy_rules_dict, initial_states, only_events, all_traces_analysis)


# TODO:aggiungere la pipe tra evento e cluster
def event_attributes_to_state(event, attribute_list, only_events):
    event = event
    if event in ('START', 'END') or attribute_list == [] or only_events:
        state = '<' + event + '>'
    else:
        # BPI-2012
        state_list = [event] + [str(attribute) for attribute in attribute_list]
        state = '<' + ' - '.join(state_list) + '>'
    return state



def state_manipulation(state, only_events):
    if not only_events:
        modified_state = '<'+(state.replace('<','').replace('>',''))+'>'
    else:
        modified_state = '<'+(state.split(' - ')[0].replace('<','').replace('>',''))+'>'  # sepsis

    return modified_state


def get_policy_rules(MDPval, only_events):
    states_max_value = {state_manipulation(row[0], only_events): row[5] for row in MDPval}
    policy_rules = {state_manipulation(row[0], only_events): [] for row in MDPval}
    for rowid, row in enumerate(MDPval):
        state = state_manipulation(MDPval[rowid, 0], only_events)
        value = MDPval[rowid, 5]
        states_max_value[state] = max(states_max_value[state], value)
    for rowid, row in enumerate(MDPval):
        state = state_manipulation(MDPval[rowid, 0], only_events)
        next_state = state_manipulation(MDPval[rowid, 2], only_events)
        value = MDPval[rowid, 5]
        max_value = states_max_value[state]
        if value >= max_value and next_state not in policy_rules[state]:
            policy_rules[state] += [next_state]
    return policy_rules


def analysis_sc3(log, policy_rules, initial_states, only_events, all_traces_analysis):
    relevant_case_number = 0
    relevant_case_total_reward = 0
    relevant_case_min_reward = 0
    relevant_case_max_reward = 0
    complementary_case_number = 0
    complementary_case_total_reward = 0
    complementary_case_min_reward = 0
    complementary_case_max_reward = 0
    relevant_case_osent = 0
    relevant_case_oaccepted = 0
    complementary_case_osent = 0
    complementary_case_oaccepted = 0
    relevant_case_id_list = []
    n_errors = 0
    # PRIZES: low: +650, medium: +1900, high: +5900
    prize_dict = {'no': 0, 'low': 650, 'medium': 1900, 'high': 5900}
    for case_index, case in enumerate(log):  # all case in log
        # if case.attributes['concept:name'] == '174105':
        #    print('here!')
        # case_solved = False
        good_start = False
        good_path = False
        reward = 0
        n_osent = 0
        n_oaccepted = 0
        case_id = case.attributes[case_id_label]
        for event_index, event in enumerate(case):  # all event in case
            # look only to complete events
            # if event[lifecycle_label].lower() == 'complete':
            # TODO: change to use reward
            try:
                attributes_list = [event[label] for label in attributes_label_list]
                amount = event[amount_label]
            except Exception as e:
                attributes_list = []
                amount = 'no'
            try:
                current_duration = event[duration_label]
            except Exception as e:
                current_duration = 0
            current_state = event_attributes_to_state(event[event_label], attributes_list, only_events)
            # first of all look if we are at the end state
            if current_state == '<END>':
                break
            # second of all, look if the first event of a good path happens
            # look if the first event of a good path happens
            # if the first event already happened then skip this step
            if not good_path:
                if current_state in initial_states:
                    good_start = True  # may be a good path
                    good_path = True
            # third of all, look if it follows the path at the next steps
            else:
                if current_state in good_next_states or all_traces_analysis:
                    # i += 1
                    good_path = True  # useless, but for clarity
                else:
                    good_path = False
            # compute rewards
            if "O_SENT" in event[event_label]:
                n_osent = 1
            else:
                n_osent = max(n_osent, 0)
            # TODO:remove prize category and use directly the amount
            if 'O_ACCEPTED' in event[event_label]:
                prize = prize_dict[amount]
                n_oaccepted = 1
            else:
                prize = 0
                n_oaccepted = max(n_oaccepted, 0)
            reward += - current_duration * cost_factor + prize
            # compute next good states from the policy rules
            # TODO: if returns a lot of errors we need to change the way we calculate clusters
            try:
                good_next_states = policy_rules[current_state]
            except Exception as e: # in the case there is not the state in the MDP model
                n_errors += 1
                print ("Error number: " + str(n_errors))
                good_next_states = []


        if good_start:
            if good_path:
                relevant_case_number += 1
                relevant_case_id_list.append(case_id)
                relevant_case_osent += n_osent
                relevant_case_oaccepted += n_oaccepted
                relevant_case_total_reward += reward
                relevant_case_max_reward = max(relevant_case_max_reward, reward)
                if relevant_case_min_reward == 0:
                    relevant_case_min_reward = reward
                relevant_case_min_reward = min(relevant_case_min_reward, reward)
            else:
                complementary_case_number += 1
                complementary_case_osent += n_osent
                complementary_case_oaccepted += n_oaccepted
                complementary_case_total_reward += reward
                complementary_case_max_reward = max(complementary_case_max_reward, reward)
                if complementary_case_min_reward == 0:
                    complementary_case_min_reward = reward
                complementary_case_min_reward = min(complementary_case_min_reward, reward)

    if relevant_case_number > 0:
        relevant_case_avg_reward = relevant_case_total_reward / relevant_case_number
    else:
        relevant_case_avg_reward = 0

    if complementary_case_number > 0:
        complementary_case_avg_reward = complementary_case_total_reward / complementary_case_number
    else:
        complementary_case_avg_reward = 0

    if relevant_case_number + complementary_case_number > 0:
        all_case_avg_reward = (relevant_case_total_reward + complementary_case_total_reward) /\
                              (relevant_case_number + complementary_case_number)
    else:
        all_case_avg_reward = 0

    print('Results:')
    print('relevant case ids:')
    print(relevant_case_id_list)
    print('all cases passing for initial state:')
    print('number:', relevant_case_number+complementary_case_number,
          ', total reward:', relevant_case_total_reward + complementary_case_total_reward,
          ', avg reward:', all_case_avg_reward,
          ', min reward:', min(relevant_case_min_reward, complementary_case_min_reward),
          ', max reward:', max(relevant_case_max_reward, complementary_case_max_reward),
          ', n O_SENT:', relevant_case_osent + complementary_case_osent,
          ', n O_ACCEPTED:', relevant_case_oaccepted + complementary_case_oaccepted)
    print('relevant cases:')
    print('number:', relevant_case_number,
          ', total reward:', relevant_case_total_reward,
          ', avg reward:', relevant_case_avg_reward,
          ', min reward:', relevant_case_min_reward,
          ', max reward:', relevant_case_max_reward,
          ', n O_SENT:', relevant_case_osent,
          ', n O_ACCEPTED:', relevant_case_oaccepted)
    print('complementary cases:')
    print('number:', complementary_case_number,
          ', total reward:', complementary_case_total_reward,
          ', avg reward:', complementary_case_avg_reward,
          ', min reward:', complementary_case_min_reward,
          ', max reward:', complementary_case_max_reward,
          ', n O_SENT:', complementary_case_osent,
          ', n O_ACCEPTED:', complementary_case_oaccepted)

    print('')
    print('table:')
    print('n cases,avg reward,tot reward,min reward,max reward,n O_SENT,n O_ACCEPTED')
    all_list = [relevant_case_number + complementary_case_number]
    all_list += [all_case_avg_reward]
    all_list += [relevant_case_total_reward + complementary_case_total_reward]
    all_list += [min(relevant_case_min_reward, complementary_case_min_reward)]
    all_list += [max(relevant_case_max_reward, complementary_case_max_reward)]
    all_list += [relevant_case_osent + complementary_case_osent]
    all_list += [relevant_case_oaccepted + complementary_case_oaccepted]
    print(','.join([str(round(x,rounding)) for x in all_list]))
    # n cases	avg reward	tot reward	min reward	max reward	n Payment full	n Payment full < 7
    opt_list = [relevant_case_number]
    opt_list += [relevant_case_avg_reward]
    opt_list += [relevant_case_total_reward]
    opt_list += [relevant_case_min_reward]
    opt_list += [relevant_case_max_reward]
    opt_list += [relevant_case_osent]
    opt_list += [relevant_case_oaccepted]
    print(','.join([str(round(x,rounding)) for x in opt_list]))
    # n cases	avg reward	tot reward	min reward	max reward	n Payment full	n Payment full < 7
    compl_list = [complementary_case_number]
    compl_list += [complementary_case_avg_reward]
    compl_list += [complementary_case_total_reward]
    compl_list += [complementary_case_min_reward]
    compl_list += [complementary_case_max_reward]
    compl_list += [complementary_case_osent]
    compl_list += [complementary_case_oaccepted]
    print(','.join([str(round(x,rounding)) for x in compl_list]))

    # output_file = os.path.join("..", "output_data", "BPM_Scenarios", "Scenario_3-BPI_2012", "v1", 'performance', 'Scenario_3_BPI_2012_good_traces.csv')
    # with open(output_file, 'w') as f:
    #     for item in relevant_case_id_list:
    #         f.write("%s\n" % item)



if __name__ == "__main__":
    main()
