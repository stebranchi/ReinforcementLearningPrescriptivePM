import os
import pandas as pd
import numpy as np
from datetime import datetime

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def main():
    # import log
    log_path = os.path.join("..", "input_data", "RL_test", "simple_model+changes.xes")
    log = xes_importer.apply(log_path)

    # define prefix and suffix with (activity, resource) tuples
    prefix_event_resource = [('START', ''), ('A', 'Clerk-000001'), ('C', 'Sara-000004')]
    suffix_event_resource = [('E', 'Clerk-000003'), ('G', 'Sara-000002')]

    # every tuples (activity, resource) will define a state through the function event_attribute_to_state
    prefix_states = [event_attribute_to_state(event, resource) for event, resource in prefix_event_resource]
    suffix_states = [event_attribute_to_state(event, resource) for event, resource in suffix_event_resource]

    # do the analysis looking for all traces with given prefix
    # among these it separates those following or not given suffix
    # computes and prints execution time results for the two groups
    analysis0(log, prefix_states, suffix_states)


def event_attribute_to_state(event, attribute):
    event = event.upper()
    if event in ('START', 'END'):
        state = '<' + event + '>'
    else:
        attribute_clean = attribute.replace('-','').replace('0','').upper()
        state = '<' + event + ' - ' + attribute_clean + '>'
    return state


def analysis0(log, prefix_states, suffix_states):
    # output variables
    relevant_case_number = 0
    relevant_case_total_time = 0
    relevant_case_min_time = 0
    relevant_case_max_time = 0
    complementary_case_number = 0
    complementary_case_total_time = 0
    complementary_case_min_time = 0
    complementary_case_max_time = 0
    for case_index, case in enumerate(log):  # all case in log
        good_prefix = True
        good_suffix = True
        i = 0
        for event_index, event in enumerate(case):  # all event in case
            # look only to complete events
            if event['lifecycle:transition'] == 'complete':
                current_state = event_attribute_to_state(event['concept:name'], event['Resource'])

                if i == 0:
                    start_timestamp = datetime.timestamp(event['time:timestamp']) # start timer at first state

                if i < len(prefix_states) and good_prefix:
                    if current_state != prefix_states[i]: # look if trace follows prefix
                        good_prefix = False
                elif i < len(prefix_states + suffix_states) and good_prefix:
                    end_timestamp = datetime.timestamp(event['time:timestamp']) # end timer at state in position len(prefix)+len(suffix)-1
                    if current_state != suffix_states[i-len(prefix_states)]: # look if trace follows suffix
                        good_suffix = False
                else:
                    break

                i += 1

        if good_prefix:
            execution_time = end_timestamp - start_timestamp
            if good_suffix: # these are all traces following both prefix and suffix
                relevant_case_number += 1
                relevant_case_total_time += execution_time
                relevant_case_max_time = max(relevant_case_max_time, execution_time)
                if relevant_case_min_time == 0:
                    relevant_case_min_time = execution_time
                relevant_case_min_time = min(relevant_case_min_time, execution_time)
            else:  # these are all traces following prefix but not suffix
                complementary_case_number += 1
                complementary_case_total_time += execution_time
                complementary_case_max_time = max(complementary_case_max_time, execution_time)
                if complementary_case_min_time == 0:
                    complementary_case_min_time = execution_time
                complementary_case_min_time = min(complementary_case_min_time, execution_time)

    # computes avg
    if relevant_case_number > 0:
        relevant_case_avg_time = relevant_case_total_time / relevant_case_number
    else:
        relevant_case_avg_time = 0

    if complementary_case_number > 0:
        complementary_case_avg_time = complementary_case_total_time / complementary_case_number
    else:
        complementary_case_avg_time = 0

    # print results
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
