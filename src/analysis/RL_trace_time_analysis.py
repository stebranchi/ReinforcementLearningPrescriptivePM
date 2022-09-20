import os
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta

from pm4py.objects.log.importer.xes import importer as xes_importer
# from pm4py.objects.log.exporter.xes import exporter as xes_exporter

event_label = 'concept:name'
lifecycle_label = 'lifecycle:transition'
timestamp_label = 'time:timestamp'
# resource_label = 'Resource' # simple model
# resource_label = 'org:group' # sepsis
resource_label = 'org:resource' # BPI_2012

def main():
    # import log
    # log_path = os.path.join("..", "input_data", "RL_test", "simple_model+changes_2.xes")
    # log_path = os.path.join("..", "input_data", "sepsis", "log", "Sepsis Cases - Event Log_testing_w_ends.xes")
    log_path = os.path.join("..", "data", "BPI2013", "BPI_2012_log_eng_preprocessed_filtered.xes")
    output_file = os.path.join("..", "data", "BPI2013", "BPI_2012_test_trace_time_delta.csv")
    log = xes_importer.apply(log_path)

    list_time_delta = []
    for case_index, case in enumerate(log):
        start_current_case = case._list[0][timestamp_label]
        if case_index > 0:
            time_delta = start_current_case - start_previous_case
            list_time_delta += [time_delta.total_seconds()]
        start_previous_case = start_current_case

    avg_time_delta = sum(list_time_delta)/len(list_time_delta)
    min_time_delta = min(list_time_delta)
    max_time_delta = max(list_time_delta)
    print('time delta:')
    print('avg:', avg_time_delta)
    print('min:', min_time_delta)
    print('max:', max_time_delta)

    with open(output_file, 'w') as f:
        for item in list_time_delta:
            f.write("%s\n" % item)








if __name__ == "__main__":
    main()
