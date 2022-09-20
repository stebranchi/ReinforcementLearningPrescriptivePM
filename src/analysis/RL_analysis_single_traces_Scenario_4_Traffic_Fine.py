import os
import pandas as pd
import numpy as np
from datetime import datetime
from csv import DictWriter

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

trace_id_label = 'concept:name'
case_id_label = 'concept:name'
event_label = 'concept:name'
lifecycle_label = 'lifecycle:transition'
timestamp_label = 'time:timestamp'
attributes_label_list = ['amount_category']
amount_category_label = 'amount_category'
two_months_label = 'two_months_passed'

def main():
    # import log
    # log_path = os.path.join("..", "input_data", "BPM_Scenarios", "Scenario_4-Traffic_Fine", "sc4_v1", "log", "Road_Traffic_Fine_training_60_mid_preprocessed.xes")
    log_path = "Road_Traffic_Fine_testing_40_mid_preprocessed.xes"
    log = xes_importer.apply(log_path)

    full_prefix = True  # False: consider only final state of the prefix to take optimal continuations
    only_events = False  # means no resources are used in definition of states

    # import policy csv, must have these columns (s,a,s',p,r,q)
    policy_file = "Trimmed Traffic_Fine mdp training 60 policy.csv"
    # policy_file = "Trimmed Traffic_Fine mdp training 60 r2 policy.csv"

    MDP_policy = pd.read_csv(policy_file)
    MDP_policy_val = MDP_policy.values  # converts it into array
    policy_rules_dict = get_policy_rules(MDP_policy_val, only_events)  # policy rules give a list of possible next state for each current state

    # output file path
    output_file_path = "performance_single_trace_testing_40.csv"
    once = True  # variable used to truncate and write header of csv

    # dictionary with info on opt paths
    trace_analysis_dict = {}

    # max_len_all_cases = max([len(case) for case in log])
    for case_index, case in enumerate(log):
        print(len(log)-case_index)
        # does the prediction at every prefix length
        trace_id = case.attributes[trace_id_label]
        trace_len = len(case)
        prefix = ''
        for length in range(len(case)):
            # last event and resource of the prefix is the start state for the predictive model
            start_event = case[length][event_label]
            try:
                start_attribute_list = [case[length][attribute] for attribute in attributes_label_list]
            except Exception as e:
                start_attribute_list = []
            initial_state = event_attributes_to_state(start_event, start_attribute_list, only_events)
            if full_prefix:
                prefix += initial_state

            # # actual duration is computed as different from timestamps of last prefix event and last trace event
            # start_timestamp = datetime.timestamp(case[length][timestamp_label])
            # end_timestamp = datetime.timestamp(case[trace_len-1][timestamp_label])
            # actual_duration = end_timestamp - start_timestamp

            # actual rewards
            actual_reward1 = 0
            actual_reward3 = 0
            actual_payment_full_6 = 0
            actual_payment_full_7 = 0
            for i in range(0, trace_len):
                try:
                    two_month_period = case[i][two_months_label]
                except Exception as e:
                    two_month_period = 0
                actual_reward1 = update_reward(case[i][event_label], two_month_period, 1, actual_reward1)
                actual_reward3 = update_reward(case[i][event_label], two_month_period, 3, actual_reward3)
            if actual_reward3 > 1:
                actual_payment_full_6 = 1
            if actual_reward1 == 1:
                actual_payment_full_7 = 1

            # analysis_tr_sc4 get the avg duration and reward for every trace which contains initial state and follows optimal path
            # if no such trace is found then it gives back None

            if full_prefix:
                if prefix not in trace_analysis_dict.keys():
                    trace_analysis_dict[prefix] = analysis_tr_sc4(log, policy_rules_dict, initial_state, only_events, full_prefix, prefix)
                opt_trace_number, opt_trace_rew1, opt_trace_rew3, opt_trace_pay, opt_trace_pay_6, \
                compl_trace_number, compl_trace_rew1, compl_trace_rew3, compl_trace_pay, compl_trace_pay_6 = trace_analysis_dict[prefix]

            else:
                if initial_state not in trace_analysis_dict.keys():
                    trace_analysis_dict[initial_state] = analysis_tr_sc4(log, policy_rules_dict, initial_state, only_events, full_prefix, prefix)
                opt_trace_number, opt_trace_rew1, opt_trace_rew3, opt_trace_pay, opt_trace_pay_6, \
                compl_trace_number, compl_trace_rew1, compl_trace_rew3, compl_trace_pay, compl_trace_pay_6 = trace_analysis_dict[initial_state]

            # print output as a test, it will be replaced with insert row in csv
            # print("trace_id:", trace_id, ", trace length:", trace_len, ", prefix length:", length + 1, ", initial_state:" , initial_state[0],
            #       ", num opt traces:", opt_trace_num, ", avg time opt:", opt_trace_avg_time, ", actual time:", actual_duration)

            # line_to_print = dict()
            # line_to_print['trace_id'] = trace_id
            # line_to_print['trace_len'] = trace_len
            # line_to_print['prefix_len'] = length + 1
            # line_to_print['initial_state'] = initial_state
            # line_to_print['opt_trace_num'] = opt_trace_num
            # line_to_print['opt_avg_time'] = opt_trace_avg_time
            # line_to_print['trace_actual_time'] = actual_duration
            # line_to_print['opt_accepted_rate'] = opt_trace_accepted_rate
            # line_to_print['compl_accepted_rate'] = compl_trace_acceptance_rate
            # line_to_print['actual_accepted'] = actual_accepted
            # line_to_print['opt_avg_prize'] = opt_trace_avg_prize
            # line_to_print['actual_prize'] = actual_prize

            line_to_print = dict()
            line_to_print['trace_id'] = trace_id
            line_to_print['trace_len'] = trace_len
            line_to_print['prefix_len'] = length + 1
            line_to_print['suffix_len'] = trace_len - length - 1
            line_to_print['initial_state'] = initial_state
            line_to_print['opt_trace_num'] = opt_trace_number
            line_to_print['opt_avg_rew1'] = opt_trace_rew1
            line_to_print['opt_avg_rew3'] = opt_trace_rew3
            line_to_print['opt_pay_rate'] = opt_trace_pay
            line_to_print['opt_pay_rate_6'] = opt_trace_pay_6
            line_to_print['compl_trace_num'] = compl_trace_number
            line_to_print['compl_avg_rew1'] = compl_trace_rew1
            line_to_print['compl_avg_rew3'] = compl_trace_rew3
            line_to_print['compl_pay_rate'] = compl_trace_pay
            line_to_print['compl_pay_rate_6'] = compl_trace_pay_6
            line_to_print['trace_actual_rew1'] = actual_reward1
            line_to_print['trace_actual_rew3'] = actual_reward3
            line_to_print['trace_actual_pay'] = actual_payment_full_6 + actual_payment_full_7
            line_to_print['trace_actual_pay_6'] = actual_payment_full_6

            if once and os.path.exists(output_file_path):
                os.remove(output_file_path)
            with open(output_file_path, 'a+', newline='') as output_file:
                dict_writer = DictWriter(output_file, fieldnames=list(line_to_print.keys()))
                while once:
                    output_file.truncate()
                    dict_writer.writeheader()
                    once = False
                dict_writer.writerow(line_to_print)
                output_file.close()


def update_reward(event, two_month_period, reward_type, current_reward):
    if reward_type == 1:
        if 'Appeal to Judge-G' in event and 'Complete' not in event:
            current_reward = -1
        elif 'Send Appeal to Prefecture-#' in event and 'Complete' not in event:
            current_reward = -1
        elif 'Payment full fine' in event and 'Complete' not in event:
            current_reward = max(6 - two_month_period, -1)
    elif reward_type == 2:
        if 'Appeal to Judge-G' in event and 'Complete' not in event:
            current_reward = -1
        elif 'Send Appeal to Prefecture-#' in event and 'Complete' not in event:
            current_reward = -1
        elif 'Payment full fine' in event and 'Complete' not in event and two_month_period <= 6:
            current_reward = 1
    elif reward_type == 3:
        if 'Appeal to Judge-G' in event and 'Complete' not in event:
            current_reward = -1
        elif 'Send Appeal to Prefecture-#' in event and 'Complete' not in event:
            current_reward = -1
        elif 'Payment full fine' in event and 'Complete' not in event:
            if two_month_period <= 2:
                current_reward = 3
            elif two_month_period <= 5:
                current_reward = 2
            else:
                current_reward = 1

    return current_reward


def event_attributes_to_state(event, attribute_list, only_events):
    event = event
    if event in ('START', 'END') or attribute_list == [] or only_events:
        state = '<' + event + '>'
    else:
        # Traffic_Fine <event - amount_category>
        state_list = [event] + [str(attribute) for attribute in attribute_list]
        state = '<' + ' - '.join(state_list) + '>'
    return state


def state_manipulation(state, only_events):
    if not only_events:
        modified_state = '<'+(state.replace('<','').replace('>',''))+'>'
    else:
        modified_state = '<'+(state.split(' - ')[0].replace('<','').replace('>',''))+'>'

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


def analysis_tr_sc4(log, policy_rules, initial_state, only_events, full_prefix, prefix):
    relevant_case_number = 0
    relevant_case_total_reward1 = 0
    relevant_case_total_reward3 = 0
    relevant_case_payment_full_6 = 0
    relevant_case_payment_full_7 = 0
    complementary_case_number = 0
    complementary_case_total_reward1 = 0
    complementary_case_total_reward3 = 0
    complementary_case_payment_full_6 = 0
    complementary_case_payment_full_7 = 0
    for case_index, case in enumerate(log):  # all case in log
        case_solved = False
        good_start = False
        good_path = False
        reward1 = 0
        reward3 = 0
        payment_full_6 = 0
        payment_full_7 = 0
        current_prefix = ''
        for event_index, event in enumerate(case):  # all event in case
            # look only to complete events
            if True: #event[lifecycle_label].lower() == 'complete':
                try:
                    attributes_list = [event[label] for label in attributes_label_list]
                except Exception as e:
                    attributes_list = []
                try:
                    two_month_period = int(event[two_months_label])
                except Exception as e:
                    two_month_period = 0

                current_state = event_attributes_to_state(event[event_label], attributes_list, only_events)
                current_prefix += current_state
                # first of all look if we are at the end state
                if current_state == '<END>':
                    break
                # second of all, look if the first event of a good path happens
                # if the first event already happened then skip this step
                if not good_path and full_prefix:
                    if current_prefix == prefix:
                        good_start = True  # may be a good path
                        good_path = True
                elif not good_path:
                    if current_state == initial_state:
                        good_start = True  # may be a good path
                        good_path = True
                # third of all, look if it follows the path at the next steps
                else:
                    if current_state in good_next_states:
                        # i += 1
                        good_path = True  # useless, but for clarity
                    else:
                        good_path = False
                # compute rewards
                reward1 = update_reward(event[event_label], two_month_period, 1, reward1)
                reward3 = update_reward(event[event_label], two_month_period, 3, reward3)
                if reward3 > 1:
                    payment_full_6 = 1
                if reward3 == 1:
                    payment_full_7 = 1

                # compute next good states from the policy rules
                try:
                    good_next_states = policy_rules[current_state]
                except Exception as e: # in the case there is not the state in the MDP model
                    good_next_states = []


        if good_start:
            if good_path:
                relevant_case_number += 1
                relevant_case_total_reward1 += reward1
                relevant_case_total_reward3 += reward3
                relevant_case_payment_full_6 += payment_full_6
                relevant_case_payment_full_7 += payment_full_7
            else:
                complementary_case_number += 1
                complementary_case_total_reward1 += reward1
                complementary_case_total_reward3 += reward3
                complementary_case_payment_full_6 += payment_full_6
                complementary_case_payment_full_7 += payment_full_7

    if relevant_case_number > 0:
        relevant_case_avg_reward1 = relevant_case_total_reward1 / relevant_case_number
        relevant_case_avg_reward3 = relevant_case_total_reward3 / relevant_case_number
        relevant_case_payment_full_rate = (relevant_case_payment_full_6 + relevant_case_payment_full_7) / relevant_case_number
        relevant_case_payment_full_6_rate = relevant_case_payment_full_6 / relevant_case_number
    else:
        relevant_case_avg_reward1 = None
        relevant_case_avg_reward3 = None
        relevant_case_payment_full_rate = None
        relevant_case_payment_full_6_rate = None

    if complementary_case_number > 0:
        complementary_case_avg_reward1 = complementary_case_total_reward1 / complementary_case_number
        complementary_case_avg_reward3 = complementary_case_total_reward3 / complementary_case_number
        complementary_case_payment_full_rate = (complementary_case_payment_full_6 + complementary_case_payment_full_7) / complementary_case_number
        complementary_case_payment_full_6_rate = complementary_case_payment_full_6 / complementary_case_number
    else:
        complementary_case_avg_reward1 = None
        complementary_case_avg_reward3 = None
        complementary_case_payment_full_rate = None
        complementary_case_payment_full_6_rate = None

    # print('Results:')
    # print('relevant cases:')
    # print('number:', relevant_case_number,
    #       ', total time:', relevant_case_total_time,
    #       ', avg time:', relevant_case_avg_time,
    #       ', avg prize:', relevant_case_avg_prize)
    # print('complementary cases:')
    # print('number:', complementary_case_number,
    #       ', total time:', complementary_case_total_time,
    #       ', avg time:', complementary_case_avg_time,
    #       ', avg prize:', complementary_case_avg_prize)

    return relevant_case_number, relevant_case_avg_reward1, relevant_case_avg_reward3, \
           relevant_case_payment_full_rate, relevant_case_payment_full_6_rate, \
           complementary_case_number, complementary_case_avg_reward1, complementary_case_avg_reward3, \
           complementary_case_payment_full_rate, complementary_case_payment_full_6_rate


if __name__ == "__main__":
    main()
