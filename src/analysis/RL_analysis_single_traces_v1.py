import os
import pandas as pd
import numpy as np
from datetime import datetime
from csv import DictWriter

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


trace_id_label = 'concept:name'
event_label = 'concept:name'
lifecycle_label = 'lifecycle:transition'
timestamp_label = 'time:timestamp'
# resource_label = 'org:resource' # simple model
resource_label = 'org:group' # sepsis
# resource_label = 'org:resource' # BPI_2012

def main():
    # import log
    # log_path = os.path.join("..", "input_data", "RL_test", "simple_model+changes_2_only_completed.xes")
    #log_path = os.path.join("..", "input_data", "sepsis", "log", "Sepsis Cases - Event Log_testing_w_ends.xes")
    log_path = os.path.join("..", "data", "logs", "Sepsis Cases - Event Log_testing_resources_w_ends.xes")
    # log_path = os.path.join("..", "input_data", "BPI_2012", "log", "BPI_2012_log_with_ends_test.xes")
    log = xes_importer.apply(log_path)

    only_events = False  # means no resources are used in definition of states

    # import policy csv, must have these columns (s,a,s',p,r,q)
    # policy_file = '../output_data/new_SB/SB_simple_model_MDP_p1_r1_Q.csv'
    #policy_file = os.path.join("..", "output_data", "sepsis", "Q_values", "sepsis_model_MDP_r_decl3_Q.csv")
    policy_file = os.path.join("..", "data", "output", "sepsis_model_MDP_r_MC_90%.csv")
    # policy_file = os.path.join("..", "output_data", "BPI_2012", "Q_values", "BPI_2012_MDP_r3_Q_3.csv")
    MDP_policy = pd.read_csv(policy_file)
    MDP_policy_val = MDP_policy.values  # convert into array
    policy_rules_dict = get_policy_rules(MDP_policy_val, only_events)  # policy rules give a list of possible next state for each current state

    # output file path
    output_file_path = os.path.join("..", "data", "evaluation", "performance_sepsis_model_90%_Q_event_resource.csv")
    once = True # variable used to truncate and write header of csv

    # max_len_all_cases = max([len(case) for case in log])
    for case_index, case in enumerate(log):
        print(len(log)-case_index)
        # does the prediction at every prefix length
        trace_id = case.attributes[trace_id_label]
        trace_len = len(case)
        for length in range(len(case)):
            # last event and resource of the prefix is the start state for the predictive model
            start_event = case[length][event_label]
            try:
                start_resource = case[length][resource_label]
            except Exception as e:
                start_resource = ""
            initial_states = [event_attribute_to_state(start_event, start_resource, only_events)]
            # actual duration is computed as different from timestamps of last prefix event and last trace event
            start_timestamp = datetime.timestamp(case[length][timestamp_label])
            end_timestamp = datetime.timestamp(case[trace_len-1][timestamp_label])
            actual_duration = end_timestamp - start_timestamp
            # analysis5 get the avg duration for every trace which contains initial state and follows optimal path
            # if no such trace is found then it gives back None
            opt_trace_num, opt_trace_avg_time = analysis5(log, policy_rules_dict, initial_states, only_events)

            # print output as a test, it will be replaced with insert row in csv
            # print("trace_id:", trace_id, ", trace length:", trace_len, ", prefix length:", length + 1, ", initial_state:" , initial_states[0],
            #       ", num opt traces:", opt_trace_num, ", avg time opt:", opt_trace_avg_time, ", actual time:", actual_duration)

            line_to_print = dict()
            line_to_print['trace_id'] = trace_id
            line_to_print['trace_len'] = trace_len
            line_to_print['prefix_len'] = length + 1
            line_to_print['initial_state'] = initial_states[0]
            line_to_print['opt_trace_num'] = opt_trace_num
            line_to_print['opt_avg_time'] = opt_trace_avg_time
            line_to_print['trace_actual_time'] = actual_duration

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



    # events_and_resources = [('START', '')]
    # events_and_resources = [('C', 'Sara-000002')]
    # initial_states = [event_attribute_to_state(event, resource, only_events) for event, resource in events_and_resources]
    # analysis4(log, policy_rules_dict, initial_states, only_events)


def event_attribute_to_state(event, attribute, only_events):
    event = event.upper()
    if event in ('START', 'END') or attribute == "" or only_events:
        state = '<' + event + '>'
    else:
        # attribute_clean = attribute.replace('-','').replace('0','').upper() # old manual simple model
        # # state = '<' + event + ', ' + attribute_clean + '>' # old manual simple model
        attribute_clean = attribute.upper()
        # state = '<' + event + ' - ' + attribute_clean + '>' # simple_model
        state = '<' + event + '-' + attribute_clean + '>' # sepsis
        # state = '<' + event + ' - ' + attribute_clean + '>' # BPI_2012
    return state


def state_manipulation(state, only_events):
    if not only_events:
        modified_state = state.upper()
    else:
        # modified_state = state.upper().split(' - ')[0].replace('>','')+'>' # simple model
        modified_state = state.upper().split('-')[0].replace('>', '') + '>'  # sepsis
        # modified_state = state.upper().split(' - ')[0].replace('>', '') + '>'  # BPI_2012

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


def analysis5(log, policy_rules, initial_state, only_events):
    relevant_case_number = 0
    relevant_case_total_time = 0
    relevant_case_min_time = 0
    relevant_case_max_time = 0
    complementary_case_number = 0
    complementary_case_total_time = 0
    complementary_case_min_time = 0
    complementary_case_max_time = 0
    for case_index, case in enumerate(log):  # all case in log
        case_solved = False
        good_start = False
        good_path = False
        for event_index, event in enumerate(case):  # all event in case
            # look only to complete events
            if event[lifecycle_label].lower() == 'complete':
                try:
                    current_resource = event[resource_label]
                except Exception as e:
                    current_resource = ""
                current_state = event_attribute_to_state(event[event_label], current_resource, only_events)
                # first of all look if we are at the end state
                if current_state == '<END>':
                    end_timestamp = datetime.timestamp(event[timestamp_label])  # stop the timer
                    break
                # second of all, look if the first event of a good path happens
                # if the first event already happened then skip this step
                if not good_path:
                    if current_state in initial_state:
                        good_start = True  # may be a good path
                        good_path = True
                        start_timestamp = datetime.timestamp(event[timestamp_label])  # start the timer
                # third of all, look if it follows the path at the next steps
                else:
                    if current_state in good_next_states:
                        # i += 1
                        good_path = True  # useless, but for clarity
                    else:
                        good_path = False
                # compute next good states from the policy rules
                try:
                    good_next_states = policy_rules[current_state]
                except Exception as e: # in the case there is not the state in the MDP model
                    good_next_states = []


        if good_start:
            execution_time = end_timestamp - start_timestamp
            if good_path:
                relevant_case_number += 1
                relevant_case_total_time += execution_time
                relevant_case_max_time = max(relevant_case_max_time, execution_time)
                if relevant_case_min_time == 0:
                    relevant_case_min_time = execution_time
                relevant_case_min_time = min(relevant_case_min_time, execution_time)
            else:
                complementary_case_number += 1
                complementary_case_total_time += execution_time
                complementary_case_max_time = max(complementary_case_max_time, execution_time)
                if complementary_case_min_time == 0:
                    complementary_case_min_time = execution_time
                complementary_case_min_time = min(complementary_case_min_time, execution_time)

    if relevant_case_number > 0:
        relevant_case_avg_time = relevant_case_total_time / relevant_case_number
    else:
        relevant_case_avg_time = None

    if complementary_case_number > 0:
        complementary_case_avg_time = complementary_case_total_time / complementary_case_number
    else:
        complementary_case_avg_time = None

    # print('Results:')
    # print('relevant cases:')
    # print('number:', relevant_case_number,
    #       ', total time:', relevant_case_total_time,
    #       ', avg time:', relevant_case_avg_time,
    #       ', min time:', relevant_case_min_time,
    #       ', max time:', relevant_case_max_time)
    # print('complementary cases:')
    # print('number:', complementary_case_number,
    #       ', total time:', complementary_case_total_time,
    #       ', avg time:', complementary_case_avg_time,
    #       ', min time:', complementary_case_min_time,
    #       ', max time:', complementary_case_max_time)

    return relevant_case_number, relevant_case_avg_time


def analysis4(log, policy_rules, initial_state, only_events):
    relevant_case_number = 0
    relevant_case_total_time = 0
    relevant_case_min_time = 0
    relevant_case_max_time = 0
    complementary_case_number = 0
    complementary_case_total_time = 0
    complementary_case_min_time = 0
    complementary_case_max_time = 0
    for case_index, case in enumerate(log):  # all case in log
        case_solved = False
        good_start = False
        good_path = False
        for event_index, event in enumerate(case):  # all event in case
            # look only to complete events
            if event[lifecycle_label] == 'complete':
                current_state = event_attribute_to_state(event[event_label], event[resource_label], only_events)
                # first of all look if we are at the end state
                if current_state == '<END>':
                    end_timestamp = datetime.timestamp(event[timestamp_label])  # stop the timer
                    break
                # second of all, look if the first event of a good path happens
                # if the first event already happened then skip this step
                if not good_path:
                    if current_state in initial_state:
                        good_start = True  # may be a good path
                        good_path = True
                        start_timestamp = datetime.timestamp(event[timestamp_label])  # start the timer
                # third of all, look if it follows the path at the next steps
                else:
                    if current_state in good_next_states:
                        # i += 1
                        good_path = True  # useless, but for clarity
                    else:
                        good_path = False
                # compute next good states from the policy rules
                try:
                    good_next_states = policy_rules[current_state]
                except Exception as e: # in the case there is not the state in the MDP model
                    good_next_states = []


        if good_start:
            execution_time = end_timestamp - start_timestamp
            if good_path:
                relevant_case_number += 1
                relevant_case_total_time += execution_time
                relevant_case_max_time = max(relevant_case_max_time, execution_time)
                if relevant_case_min_time == 0:
                    relevant_case_min_time = execution_time
                relevant_case_min_time = min(relevant_case_min_time, execution_time)
            else:
                complementary_case_number += 1
                complementary_case_total_time += execution_time
                complementary_case_max_time = max(complementary_case_max_time, execution_time)
                if complementary_case_min_time == 0:
                    complementary_case_min_time = execution_time
                complementary_case_min_time = min(complementary_case_min_time, execution_time)

    if relevant_case_number > 0:
        relevant_case_avg_time = relevant_case_total_time / relevant_case_number
    else:
        relevant_case_avg_time = 0

    if complementary_case_number > 0:
        complementary_case_avg_time = complementary_case_total_time / complementary_case_number
    else:
        complementary_case_avg_time = 0

    print('Results:')
    print('relevant cases:')
    print('number:', relevant_case_number,
          ', total time:', relevant_case_total_time,
          ', avg time:', relevant_case_avg_time,
          ', min time:', relevant_case_min_time,
          ', max time:', relevant_case_max_time)
    print('complementary cases:')
    print('number:', complementary_case_number,
          ', total time:', complementary_case_total_time,
          ', avg time:', complementary_case_avg_time,
          ', min time:', complementary_case_min_time,
          ', max time:', complementary_case_max_time)



if __name__ == "__main__":
    main()
