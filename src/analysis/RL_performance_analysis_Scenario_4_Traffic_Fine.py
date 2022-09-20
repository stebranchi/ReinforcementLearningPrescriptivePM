import os
import pandas as pd
# import numpy as np
# from datetime import datetime
import math
from csv import DictWriter

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

def main():



    # import policy csv, must have these columns (s,a,s',p,r,q)
    performance_file_name = "performance_single_trace_testing_20_r3.csv"
    performance_path = os.path.join("..", "final_evaluation", "Fine", "performance", performance_file_name)
    performance = pd.read_csv(performance_path)
    # performance_val = performance.values  # converts it into array

    print('ciao')

    #00 line_to_print['trace_id'] = trace_id
    #01 line_to_print['trace_len'] = trace_len
    #02 line_to_print['prefix_len'] = length + 1
    #03 line_to_print['suffix_len'] = trace_len - length - 1
    #04 line_to_print['initial_state'] = initial_state
    #05 line_to_print['opt_trace_num'] = opt_trace_number
    #06 line_to_print['opt_avg_rew1'] = opt_trace_rew1
    #07 line_to_print['opt_avg_rew3'] = opt_trace_rew3
    #08 line_to_print['opt_pay_rate'] = opt_trace_pay
    #09 line_to_print['opt_pay_rate_6'] = opt_trace_pay_6
    #10 line_to_print['compl_trace_num'] = compl_trace_number
    #11 line_to_print['compl_avg_rew1'] = compl_trace_rew1
    #12 line_to_print['compl_avg_rew3'] = compl_trace_rew3
    #13 line_to_print['compl_pay_rate'] = compl_trace_pay
    #14 line_to_print['compl_pay_rate_6'] = compl_trace_pay_6
    #15 line_to_print['trace_actual_rew1'] = actual_reward1
    #16 line_to_print['trace_actual_rew3'] = actual_reward2
    #17 line_to_print['trace_actual_pay'] = actual_payment_full_6 + actual_payment_full_7
    #18 line_to_print['trace_actual_pay_6'] = actual_payment_full_6

    prefix_min = min(list(performance['prefix_len']))
    prefix_max = max(list(performance['prefix_len']))
    suffix_min = min(list(performance['suffix_len']))
    suffix_max = max(list(performance['suffix_len']))

    prefix_count_opt_rew1_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max+1)}
    suffix_count_opt_rew1_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max+1)}
    prefix_count_opt_rew3_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max + 1)}
    suffix_count_opt_rew3_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max + 1)}
    prefix_count_compl_rew1_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max+1)}
    suffix_count_compl_rew1_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max+1)}
    prefix_count_compl_rew3_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max + 1)}
    suffix_count_compl_rew3_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max + 1)}
    prefix_opt_rew1_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max+1)}
    suffix_opt_rew1_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max+1)}
    prefix_opt_rew1_avg_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max+1)}
    suffix_opt_rew1_avg_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max+1)}
    prefix_compl_rew1_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max+1)}
    suffix_compl_rew1_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max+1)}
    prefix_compl_rew1_avg_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max+1)}
    suffix_compl_rew1_avg_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max+1)}
    prefix_act_rew1_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max+1)}
    suffix_act_rew1_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max+1)}
    prefix_act_rew1_avg_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max+1)}
    suffix_act_rew1_avg_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max+1)}
    prefix_opt_rew3_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max + 1)}
    suffix_opt_rew3_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max + 1)}
    prefix_opt_rew3_avg_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max + 1)}
    suffix_opt_rew3_avg_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max + 1)}
    prefix_compl_rew3_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max + 1)}
    suffix_compl_rew3_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max + 1)}
    prefix_compl_rew3_avg_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max + 1)}
    suffix_compl_rew3_avg_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max + 1)}
    prefix_act_rew3_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max + 1)}
    suffix_act_rew3_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max + 1)}
    prefix_act_rew3_avg_dict = {prefix: 0 for prefix in range(prefix_min, prefix_max + 1)}
    suffix_act_rew3_avg_dict = {suffix: 0 for suffix in range(suffix_min, suffix_max + 1)}

    for i in range(len(performance)):
        prefix = performance['prefix_len'][i]
        suffix = performance['suffix_len'][i]
        if not math.isnan(performance['opt_avg_rew1'][i]):
            prefix_count_opt_rew1_dict[prefix] += 1
            suffix_count_opt_rew1_dict[suffix] += 1
            prefix_opt_rew1_dict[prefix] += float(performance['opt_avg_rew1'][i])
            suffix_opt_rew1_dict[suffix] += float(performance['opt_avg_rew1'][i])
            prefix_act_rew1_dict[prefix] += float(performance['trace_actual_rew1'][i])
            suffix_act_rew1_dict[suffix] += float(performance['trace_actual_rew1'][i])
        if not math.isnan(performance['opt_avg_rew3'][i]):
            prefix_count_opt_rew3_dict[prefix] += 1
            suffix_count_opt_rew3_dict[suffix] += 1
            prefix_opt_rew3_dict[prefix] += float(performance['opt_avg_rew3'][i])
            suffix_opt_rew3_dict[suffix] += float(performance['opt_avg_rew3'][i])
            prefix_act_rew3_dict[prefix] += float(performance['trace_actual_rew3'][i])
            suffix_act_rew3_dict[suffix] += float(performance['trace_actual_rew3'][i])
        if not math.isnan(performance['compl_avg_rew1'][i]):
            prefix_count_compl_rew1_dict[prefix] += 1
            suffix_count_compl_rew1_dict[suffix] += 1
            prefix_compl_rew1_dict[prefix] += float(performance['compl_avg_rew1'][i])
            suffix_compl_rew1_dict[suffix] += float(performance['compl_avg_rew1'][i])
        if not math.isnan(performance['compl_avg_rew3'][i]):
            prefix_count_compl_rew3_dict[prefix] += 1
            suffix_count_compl_rew3_dict[suffix] += 1
            prefix_compl_rew3_dict[prefix] += float(performance['compl_avg_rew3'][i])
            suffix_compl_rew3_dict[suffix] += float(performance['compl_avg_rew3'][i])

    for prefix in prefix_count_opt_rew1_dict.keys():
        if prefix_count_opt_rew1_dict[prefix] > 0:
            prefix_opt_rew1_avg_dict[prefix] = prefix_opt_rew1_dict[prefix]/prefix_count_opt_rew1_dict[prefix]
            prefix_act_rew1_avg_dict[prefix] = prefix_act_rew1_dict[prefix]/prefix_count_opt_rew1_dict[prefix]
        if prefix_count_opt_rew3_dict[prefix] > 0:
            prefix_opt_rew3_avg_dict[prefix] = prefix_opt_rew3_dict[prefix]/prefix_count_opt_rew3_dict[prefix]
            prefix_act_rew3_avg_dict[prefix] = prefix_act_rew3_dict[prefix]/prefix_count_opt_rew3_dict[prefix]
        if prefix_count_compl_rew1_dict[prefix] > 0:
            prefix_compl_rew1_avg_dict[prefix] = prefix_compl_rew1_dict[prefix]/prefix_count_compl_rew1_dict[prefix]
        if prefix_count_compl_rew3_dict[prefix] > 0:
            prefix_compl_rew3_avg_dict[prefix] = prefix_compl_rew3_dict[prefix]/prefix_count_compl_rew3_dict[prefix]

    for suffix in suffix_count_opt_rew1_dict.keys():
        if suffix_count_opt_rew1_dict[suffix] > 0:
            suffix_opt_rew1_avg_dict[suffix] = suffix_opt_rew1_dict[suffix]/suffix_count_opt_rew1_dict[suffix]
            suffix_act_rew1_avg_dict[suffix] = suffix_act_rew1_dict[suffix]/suffix_count_opt_rew1_dict[suffix]
        if suffix_count_opt_rew3_dict[suffix] > 0:
            suffix_opt_rew3_avg_dict[suffix] = suffix_opt_rew3_dict[suffix]/suffix_count_opt_rew3_dict[suffix]
            suffix_act_rew3_avg_dict[suffix] = suffix_act_rew3_dict[suffix]/suffix_count_opt_rew3_dict[suffix]
        if suffix_count_compl_rew1_dict[suffix] > 0:
            suffix_compl_rew1_avg_dict[suffix] = suffix_compl_rew1_dict[suffix]/suffix_count_compl_rew1_dict[suffix]
        if suffix_count_compl_rew3_dict[suffix] > 0:
            suffix_compl_rew3_avg_dict[suffix] = suffix_compl_rew3_dict[suffix]/suffix_count_compl_rew3_dict[suffix]

    print(prefix_opt_rew1_avg_dict)
    print(';'.join([str(v) for v in prefix_opt_rew1_avg_dict.values()]))
    print(suffix_opt_rew1_avg_dict)
    print(';'.join([str(v) for v in suffix_opt_rew1_avg_dict.values()]))

    once = True
    output_file_name = 'avg_' + performance_file_name
    output_file_path = os.path.join("..", "final_evaluation", "Fine", "performance_out", output_file_name)
    for i in range(len(prefix_count_opt_rew1_dict.keys())):
        line_to_print = dict()
        line_to_print['prefix_len'] = list(prefix_count_opt_rew1_dict.keys())[i]
        line_to_print['p count opt rew1'] = list(prefix_count_opt_rew1_dict.values())[i]
        line_to_print['p opt rew1'] = list(prefix_opt_rew1_avg_dict.values())[i]
        line_to_print['p act rew1'] = list(prefix_act_rew1_avg_dict.values())[i]
        line_to_print['p delta rew1'] = list(prefix_opt_rew1_avg_dict.values())[i]-list(prefix_act_rew1_avg_dict.values())[i]
        line_to_print['p count compl rew1'] = list(prefix_count_compl_rew1_dict.values())[i]
        line_to_print['p compl rew1'] = list(prefix_compl_rew1_avg_dict.values())[i]
        line_to_print['p count opt rew3'] = list(prefix_count_opt_rew3_dict.values())[i]
        line_to_print['p opt rew3'] = list(prefix_opt_rew3_avg_dict.values())[i]
        line_to_print['p act rew3'] = list(prefix_act_rew3_avg_dict.values())[i]
        line_to_print['p delta rew3'] = list(prefix_opt_rew3_avg_dict.values())[i]-list(prefix_act_rew3_avg_dict.values())[i]
        line_to_print['p count compl rew3'] = list(prefix_count_compl_rew3_dict.values())[i]
        line_to_print['p compl rew3'] = list(prefix_compl_rew3_avg_dict.values())[i]
        line_to_print[''] = ''
        line_to_print['suffix_len'] = list(suffix_count_opt_rew1_dict.keys())[i]
        line_to_print['s count opt rew1'] = list(suffix_count_opt_rew1_dict.values())[i]
        line_to_print['s opt rew1'] = list(suffix_opt_rew1_avg_dict.values())[i]
        line_to_print['s act rew1'] = list(suffix_act_rew1_avg_dict.values())[i]
        line_to_print['s delta rew1'] = list(suffix_opt_rew1_avg_dict.values())[i]-list(suffix_act_rew1_avg_dict.values())[i]
        line_to_print['s count compl rew1'] = list(suffix_count_compl_rew1_dict.values())[i]
        line_to_print['s compl rew1'] = list(suffix_compl_rew1_avg_dict.values())[i]
        line_to_print['s count opt rew3'] = list(suffix_count_opt_rew3_dict.values())[i]
        line_to_print['s opt rew3'] = list(suffix_opt_rew3_avg_dict.values())[i]
        line_to_print['s act rew3'] = list(suffix_act_rew3_avg_dict.values())[i]
        line_to_print['s delta rew3'] = list(suffix_opt_rew3_avg_dict.values())[i]-list(suffix_act_rew3_avg_dict.values())[i]
        line_to_print['s count compl rew3'] = list(suffix_count_compl_rew3_dict.values())[i]
        line_to_print['s compl rew3'] = list(suffix_compl_rew3_avg_dict.values())[i]


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


if __name__ == "__main__":
    main()
