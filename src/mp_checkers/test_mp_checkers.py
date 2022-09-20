from src.parsers import *
from src.mp_checkers.log import *

from pm4py.objects.log.importer.xes import factory as xes_import_factory
log = xes_import_factory.apply(IN_LOG_PATH)


def run_all_mp_checkers():
    input = parse_decl(IN_DECL_PATH)
    activities = input.activities
    trace_set = set()
    for trace in log:
        trace_set.add(trace.attributes["concept:name"])
    for key, conditions in input.checkers.items():
        if key.startswith(EXISTENCE):
            trace_set = trace_set.intersection(mp_existence_checker(log, True, activities[0], conditions[0], conditions[1]))
        elif key.startswith(ABSENCE):
            trace_set = trace_set.intersection(mp_absence_checker(log, True, activities[0], conditions[0], conditions[1]))
        elif key.startswith(INIT):
            trace_set = trace_set.intersection(mp_init_checker(log, True, activities[0], conditions[0]))
        elif key.startswith(EXACTLY):
            trace_set = trace_set.intersection(mp_exactly_checker(log, True, activities[0], conditions[0], conditions[1]))
        elif key.startswith(CHOICE):
            trace_set = trace_set.intersection(mp_choice_checker(log, True, activities[0], activities[1], conditions[0]))
        elif key.startswith(EXCLUSIVE_CHOICE):
            trace_set = trace_set.intersection(mp_exclusive_choice_checker(log, True, activities[0], activities[1], conditions[0]))
        elif key.startswith(RESPONDED_EXISTENCE):
            trace_set = trace_set.intersection(mp_responded_existence_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(RESPONSE):
            trace_set = trace_set.intersection(mp_response_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(ALTERNATE_RESPONSE):
            trace_set = trace_set.intersection(mp_alternate_response_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(CHAIN_RESPONSE):
            trace_set = trace_set.intersection(mp_chain_response_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(PRECEDENCE):
            trace_set = trace_set.intersection(mp_precedence_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(ALTERNATE_PRECEDENCE):
            trace_set = trace_set.intersection(mp_alternate_precedence_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(CHAIN_PRECEDENCE):
            trace_set = trace_set.intersection(mp_chain_precedence_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(ALTERNATE_RESPONSE):
            trace_set = trace_set.intersection(mp_not_responded_existence_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(NOT_RESPONSE):
            trace_set = trace_set.intersection(mp_not_response_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(NOT_CHAIN_RESPONSE):
            trace_set = trace_set.intersection(mp_not_chain_response_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(NOT_PRECEDENCE):
            trace_set = trace_set.intersection(mp_not_precedence_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        elif key.startswith(NOT_CHAIN_PRECEDENCE):
            trace_set = trace_set.intersection(mp_not_chain_precedence_checker(log, True, activities[0], activities[1], conditions[0], conditions[1]))
        print(trace_set)

        if len(trace_set) > 0:
            return True
        else:
            return False


run_all_mp_checkers()
