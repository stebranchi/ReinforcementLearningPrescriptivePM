from src.enums import TraceSpeed
from src.constants import *
from src.mp_checkers.trace import *
from itertools import combinations

from pm4py.objects.log.importer.xes import factory as xes_import_factory
log = xes_import_factory.apply(IN_LOG_PATH)


def check_trace_speed(trace, time_threshold):
    time_diff = (
        trace[len(trace) - 1]["time:timestamp"] - trace[0]["time:timestamp"]
    ).total_seconds()
    if time_diff < time_threshold:
        result = TraceSpeed.FAST
    else:
        result = TraceSpeed.SLOW
    return result


def get_log_speed(log, time_threshold):
    result = []
    for trace in log:
        result.append(check_trace_speed(trace, time_threshold).value)
    return result


def get_num_traces_by_two_events(log, a, b):
    num_traces_satisfied = 0
    for trace in log:
        a_exists = False
        b_exists = False
        for event in trace:
            if not a_exists and event["concept:name"] == a:
                a_exists = True
            elif not b_exists and event["concept:name"] == b:
                b_exists = True
            if a_exists and b_exists:
                break
        if a_exists and b_exists:
            num_traces_satisfied += 1
    return num_traces_satisfied


# a-priori algorithm
# Description:
# pairs of events and their support (the % of traces where the pair of events occurs)
def a_priori(log):
    num_traces = len(log)
    distinct_events = set()
    result = {}
    for trace in log:
        for event in trace:
            distinct_events.add(event["concept:name"])
    pairs = list(combinations(distinct_events, 2))
    for pair in pairs:
        result[pair] = get_num_traces_by_two_events(log, pair[0], pair[1]) / num_traces
    return result


def encode_trace(log, probability_threshold, activation_rules, correlation_rules):
    frequent_pairs = [*{k:v for (k,v) in a_priori(log).items() if v > probability_threshold}]
    pairs = []
    for pair in frequent_pairs:
        (x, y) = pair
        reverse_pair = (y, x)
        pairs.extend([pair, reverse_pair])
    log_result = {}
    features = []
    encoded_data = []
    for trace in log:
        trace_result = {}
        for (a, b) in pairs:
            trace_result[RESPONDED_EXISTENCE + "[" + a + "," + b+ "]"] = mp_responded_existence_checker(trace, True, a, b, activation_rules, correlation_rules).state.value
            trace_result[RESPONSE + "[" + a + "," + b+ "]"] = mp_response_checker(trace, True, a, b, activation_rules, correlation_rules).state.value
            trace_result[ALTERNATE_RESPONSE + "[" + a + "," + b+ "]"] = mp_alternate_response_checker(trace, True, a, b, activation_rules, correlation_rules).state.value
            trace_result[CHAIN_RESPONSE + "[" + a + "," + b+ "]"] = mp_chain_response_checker(trace, True, a, b, activation_rules, correlation_rules).state.value
            trace_result[PRECEDENCE + "[" + a + "," + b+ "]"] = mp_precedence_checker(trace, True, a, b, activation_rules, correlation_rules).state.value
            trace_result[ALTERNATE_PRECEDENCE + "[" + a + "," + b+ "]"] = mp_alternate_precedence_checker(trace, True, a, b, activation_rules, correlation_rules).state.value
            trace_result[CHAIN_PRECEDENCE + "[" + a + "," + b+ "]"] = mp_chain_precedence_checker(trace, True, a, b, activation_rules, correlation_rules).state.value
        if not features:
            features = list(trace_result.keys())
        encoded_data.append(list(trace_result.values()))
        log_result[trace.attributes["concept:name"]] = trace_result
    return (features, encoded_data)