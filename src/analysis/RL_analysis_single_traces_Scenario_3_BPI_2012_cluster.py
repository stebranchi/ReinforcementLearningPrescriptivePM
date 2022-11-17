import os
import pandas as pd
import numpy as np
from datetime import datetime
from csv import DictWriter

from pm4py.objects.log.importer.xes import importer as xes_importer

q_column_index = 6
trace_id_label = 'concept:name'
case_id_label = 'concept:name'
event_label = 'concept:name'
lifecycle_label = 'lifecycle:transition'
timestamp_label = 'time:timestamp'
attributes_label_list = ['cluster:prefix']
amount_label = 'amount'
reward_label = 'kpi:reward'
duration_label = 'duration'
cost_factor = 0.005
check_single_trace = True  # analysis of one trace_id

def main():
    # import log
    # log_path = os.path.join("..", "input_data", "BPM_Scenarios", "Scenario_3-BPI_2012", "v3", "log", "BPI_2012_log_eng_training_60_mid_preprocessed_var90.xes")
    log_path =os.path.join("..", "..", "cluster_data", "output_logs", "BPI2012_log_eng_positional_cumulative_squashed_testing_20_mid_preprocessed.xes")
    log = xes_importer.apply(log_path)
    # TODO: rephrase the comment below
    full_prefix = True  # False: consider only final state of the prefix to take optimal continuations
    only_events = False  # means no resources are used in definition of states

    # import policy csv, must have these columns (s,a,s',p,r,q)
    policy_file = os.path.join("..", "..", "cluster_data", "output_policies", "BPI2012_log_eng_positional_end_reward_95.csv")

    MDP_policy = pd.read_csv(policy_file)
    MDP_policy_val = MDP_policy.values  # converts it into array
    policy_rules_dict = get_policy_rules(MDP_policy_val, only_events)  # policy rules give a list of possible next state for each current state

    # output file path
    output_file_name = "single_trace_testing_20_positional_cumulative.csv"
    output_file_path = os.path.join("..", "..", "cluster_data", "output_performance", output_file_name)


    once = True  # variable used to truncate and write header of csv

    # dictionary with info on opt paths
    trace_analysis_dict = {}

    # max_len_all_cases = max([len(case) for case in log])
    prize_dict = {'no': 0, 'low': 650, 'medium': 1900, 'high': 5900}
    for case_index, case in enumerate(log):
        print(len(log)-case_index)
        # does the prediction at every prefix length
        trace_id = case.attributes[trace_id_label]
        trace_len = len(case)
        prefix = ''
        for length in range(len(case)):
            # last event and resource of the prefix is the start state for the predictive model
            # check single prefix
            # if check_single_trace and (trace_id != '198173' or length != 0):
            #    continue
            # elif check_single_trace:
            #    print('here')

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

            # actual working-time duration is computed from event attribute duration
            actual_duration = 0
            actual_accepted = 0
            actual_prize = 0
            number_events_duration = 0
            for i in range(trace_len):
                try:
                    actual_duration += case[i][duration_label]
                    if case[i][duration_label] > 0:
                        number_events_duration += 1
                except Exception as e:
                    actual_duration += 0
                if 'O_ACCEPTED' in case[i][event_label]:
                    amount = case[i][amount_label]
                    actual_accepted = 1
                    actual_prize = int(amount)


            # analysis_tr_sc3 get the avg duration and reward for every trace which contains initial state and follows optimal path
            # if no such trace is found then it gives back None

            if full_prefix:
                if prefix not in trace_analysis_dict.keys():
                    trace_analysis_dict[prefix] = analysis_tr_sc3(log, policy_rules_dict, initial_state, only_events, full_prefix, prefix)
                opt_trace_number, opt_trace_avg_time, opt_trace_accepted_rate, opt_trace_avg_prize, \
                    compl_trace_number, compl_trace_avg_time, compl_trace_acceptance_rate, compl_trace_avg_prize = trace_analysis_dict[prefix]

            else:
                if initial_state not in trace_analysis_dict.keys():
                    trace_analysis_dict[initial_state] = analysis_tr_sc3(log, policy_rules_dict, initial_state, only_events, full_prefix, prefix)
                opt_trace_number, opt_trace_avg_time, opt_trace_accepted_rate, opt_trace_avg_prize, \
                    compl_trace_number, compl_trace_avg_time, compl_trace_acceptance_rate, compl_trace_avg_prize = trace_analysis_dict[initial_state]

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

            try:
                opt_avg_reward = - opt_trace_avg_time * cost_factor + opt_trace_avg_prize
            except Exception as e:
                opt_avg_reward = None
            try:
                compl_avg_reward = - compl_trace_avg_time * cost_factor + compl_trace_avg_prize
            except Exception as e:
                compl_avg_reward = None
            actual_reward = - actual_duration * cost_factor + actual_prize

            # except Exception as e:
            #     print('opt_trace_avg_time', opt_trace_avg_time)
            #     print('opt_trace_avg_prize', opt_trace_avg_prize)
            #     print('compl_trace_avg_time', compl_trace_avg_time)
            #     print('compl_trace_avg_prize', compl_trace_avg_prize)
            #     print('actual_duration', actual_duration)
            #     print('actual_prize', actual_prize)


            line_to_print = dict()
            line_to_print['trace_id'] = trace_id
            line_to_print['trace_len'] = trace_len
            line_to_print['prefix_len'] = length + 1
            line_to_print['suffix_len'] = trace_len - length - 1
            line_to_print['initial_state'] = initial_state
            line_to_print['opt_trace_num'] = opt_trace_number
            line_to_print['compl_trace_num'] = compl_trace_number
            line_to_print['opt_avg_reward'] = opt_avg_reward
            line_to_print['compl_avg_reward'] = compl_avg_reward
            line_to_print['trace_actual_reward'] = actual_reward
            line_to_print['opt_accepted_rate'] = opt_trace_accepted_rate
            line_to_print['compl_accepted_rate'] = compl_trace_acceptance_rate
            line_to_print['actual_accepted'] = actual_accepted

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



def event_attributes_to_state(event, attribute_list, only_events):
    event = event
    if event == 'START' or attribute_list == [] or only_events:
        state = '<' + event + '>'
    else:
        # BPI-2012
        state_list = [str(attribute) for attribute in attribute_list]
        state = '<' + event + ' | ' + ' - '.join(state_list) + '>'
    return state


def state_manipulation(state, only_events):
    if not only_events:
        modified_state = '<'+(state.replace('<','').replace('>',''))+'>'
    else:
        modified_state = '<'+(state.split(' - ')[0].replace('<','').replace('>',''))+'>'  # sepsis

    return modified_state


def get_policy_rules(MDPval, only_events):
    states_max_value = {state_manipulation(row[0], only_events): row[q_column_index] for row in MDPval}
    policy_rules = {state_manipulation(row[0], only_events): [] for row in MDPval}
    for rowid, row in enumerate(MDPval):
        state = state_manipulation(MDPval[rowid, 0], only_events)
        value = MDPval[rowid, q_column_index]
        states_max_value[state] = max(states_max_value[state], value)
    for rowid, row in enumerate(MDPval):
        state = state_manipulation(MDPval[rowid, 0], only_events)
        next_state = state_manipulation(MDPval[rowid, 2], only_events)
        value = MDPval[rowid, q_column_index]
        max_value = states_max_value[state]
        if value >= max_value and next_state not in policy_rules[state]:
            policy_rules[state] += [next_state]
    return policy_rules


def analysis_tr_sc3(log, policy_rules, initial_state, only_events, full_prefix, prefix):
    relevant_case_number = 0
    relevant_case_total_time = 0
    relevant_case_accepted_number = 0
    relevant_case_total_prize = 0
    complementary_case_number = 0
    complementary_case_total_time = 0
    complementary_case_accepted_number = 0
    complementary_case_total_prize = 0
    relevant_case_id_list = []
    relevant_case_reward = []
    for case_index, case in enumerate(log):  # all case in log
        case_solved = False
        good_start = False
        good_path = False
        duration = 0
        prize = 0
        accepted = 0
        current_prefix = ''
        case_id = case.attributes[case_id_label]
        for event_index, event in enumerate(case):  # all event in case
            # look only to complete events
            if True: #event[lifecycle_label].lower() == 'complete':
                try:
                    attributes_list = [event[label] for label in attributes_label_list]
                except Exception as e:
                    attributes_list = []
                try:
                    current_duration = event[duration_label]
                except Exception as e:
                    current_duration = 0

                current_state = event_attributes_to_state(event[event_label], attributes_list, only_events)
                current_prefix += current_state
                # first of all look if we are at the end state
                if 'END' in current_state:
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
                if 'O_ACCEPTED' in event[event_label]:
                    reward = event[reward_label]
                    prize = reward
                    accepted = 1
                duration += current_duration
                # compute next good states from the policy rules
                try:
                    good_next_states = policy_rules[current_state]
                except Exception as e: # in the case there is not the state in the MDP model
                    print()
                    good_next_states = []


        if good_start:
            if good_path:
                if check_single_trace:
                    relevant_case_id_list.append(case_id)
                    relevant_case_reward.append(- duration * cost_factor + prize)
                relevant_case_number += 1
                relevant_case_total_time += duration
                relevant_case_accepted_number += accepted
                relevant_case_total_prize += prize
            else:
                complementary_case_number += 1
                complementary_case_total_time += duration
                complementary_case_accepted_number += accepted
                complementary_case_total_prize += prize

    if relevant_case_number > 0:
        relevant_case_avg_time = relevant_case_total_time / relevant_case_number
        relevant_case_accepted_rate = relevant_case_accepted_number / relevant_case_number
        relevant_case_avg_prize = relevant_case_total_prize / relevant_case_number
    else:
        relevant_case_avg_time = None
        relevant_case_accepted_rate = None
        relevant_case_avg_prize = None

    if complementary_case_number > 0:
        complementary_case_avg_time = complementary_case_total_time / complementary_case_number
        complementary_case_accepted_rate = complementary_case_accepted_number / complementary_case_number
        complementary_case_avg_prize = complementary_case_total_prize / complementary_case_number
    else:
        complementary_case_avg_time = None
        complementary_case_accepted_rate = None
        complementary_case_avg_prize = None

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

    if check_single_trace:
        output_name = 'relevant_trace_id_198173_amount.csv'
        output_file = os.path.join("..", "..", "cluster_data", "output_performance", output_name)
        with open(output_file, 'w') as f:
            write_list = [str(relevant_case_id_list[i])+','+str(relevant_case_reward[i])
                          for i in range(len(relevant_case_id_list))]
            for item in write_list:
                f.write("%s\n" % item)

    return relevant_case_number, relevant_case_avg_time, relevant_case_accepted_rate, relevant_case_avg_prize, \
           complementary_case_number, complementary_case_avg_time, complementary_case_accepted_rate, complementary_case_avg_prize


if __name__ == "__main__":
    main()
