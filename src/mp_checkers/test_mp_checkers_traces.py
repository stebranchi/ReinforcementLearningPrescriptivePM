import ast
from src.parsers import *
from src.mp_checkers.trace.mp_choice import *
from src.mp_checkers.trace.mp_existence import *
from src.mp_checkers.trace.mp_negative_relation import *
from src.mp_checkers.trace.mp_relation import *

from pm4py.objects.log.importer.xes import importer as xes_importer


def run_all_mp_checkers_traces(trace, decl_path, txt_path, min_point, points_dict):
    if not points_dict:
        input = parse_decl(decl_path)
        points_dict = getPoints(txt_path, input)
        #checkers = input.checkers
    checkers = [ast.literal_eval(key) for key,item in points_dict.items() if item >= min_point]
    done = True
    tot_score = 0
    for constraint in checkers:
        if constraint['key'].startswith(EXISTENCE):
            result = mp_existence_checker(trace, done, constraint['attribute'],
                                                     constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ABSENCE):
            result = mp_absence_checker(trace, done, constraint['attribute'], constraint['condition'][0],
                                                   constraint['condition'][1])
        elif constraint['key'].startswith(INIT):
            result = mp_init_checker(trace, done, constraint['attribute'], constraint['condition'][0])
        elif constraint['key'].startswith(EXACTLY):
            result = mp_exactly_checker(trace, done, constraint['attribute'], constraint['condition'][0],
                                                   constraint['condition'][1])
        elif constraint['key'].startswith(CHOICE):
            result = mp_choice_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                  constraint['attribute'].split(', ')[1], constraint['condition'][0])
        elif constraint['key'].startswith(EXCLUSIVE_CHOICE):
            result = mp_exclusive_choice_checker(trace, done, constraint['attribute'].split(', ')[0],
            constraint['attribute'].split(', ')[1], constraint['condition'][0])
        elif constraint['key'].startswith(RESPONDED_EXISTENCE):
            result = mp_responded_existence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                               constraint['attribute'].split(', ')[1],
                                                               constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(RESPONSE):
            result = mp_response_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                    constraint['attribute'].split(', ')[1],
                                                    constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_RESPONSE):
            result = mp_alternate_response_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                              constraint['attribute'].split(', ')[1],
                                                              constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(CHAIN_RESPONSE):
            result = mp_chain_response_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                          constraint['attribute'].split(', ')[1],
                                                          constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(PRECEDENCE):
            result = mp_precedence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                      constraint['attribute'].split(', ')[1],
                                                      constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_PRECEDENCE):
            result = mp_alternate_precedence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                                constraint['attribute'].split(', ')[1],
                                                                constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(CHAIN_PRECEDENCE):
            result = mp_chain_precedence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                            constraint['attribute'].split(', ')[1],
                                                            constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_RESPONSE):
            result = mp_not_responded_existence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                                   constraint['attribute'].split(', ')[1],
                                                                   constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_RESPONSE):
            result = mp_not_response_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                        constraint['attribute'].split(', ')[1],
                                                        constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_CHAIN_RESPONSE):
            result = mp_not_chain_response_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                              constraint['attribute'].split(', ')[1],
                                                              constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_PRECEDENCE):
            result = mp_not_precedence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                          constraint['attribute'].split(', ')[1],
                                                          constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_CHAIN_PRECEDENCE):
            result = mp_not_chain_precedence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                                constraint['attribute'].split(', ')[1],
                                                                constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(CHAIN_SUCCESSION):
            result = mp_chain_precedence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                 constraint['attribute'].split(', ')[1],
                                                 constraint['condition'][0], constraint['condition'][1]) and \
                     mp_chain_response_checker(trace, done, constraint['attribute'].split(', ')[0],
                                               constraint['attribute'].split(', ')[1],
                                               constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_SUCCESSION):
            result = mp_alternate_precedence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                 constraint['attribute'].split(', ')[1],
                                                 constraint['condition'][0], constraint['condition'][1]) and \
                     mp_alternate_response_checker(trace, done, constraint['attribute'].split(', ')[0],
                                               constraint['attribute'].split(', ')[1],
                                               constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(SUCCESSION):
            result = mp_precedence_checker(trace, done, constraint['attribute'].split(', ')[0],
                                                 constraint['attribute'].split(', ')[1],
                                                 constraint['condition'][0], constraint['condition'][1]) and \
                     mp_response_checker(trace, done, constraint['attribute'].split(', ')[0],
                                               constraint['attribute'].split(', ')[1],
                                               constraint['condition'][0], constraint['condition'][1])


        if result:
            tot_score += points_dict[str(constraint)]

    return tot_score / 100, points_dict


def getPoints(txt_path, model):
    txt = open(txt_path, 'r')
    points = list()
    points_dict = dict()
    start_point = False
    for line in txt:
        if start_point:
            score = line.split('%')[0].split(' ')[-1]
            points.append(float(score))

        if line.startswith('Constraints:'):
            start_point = True

    for i in range(len(model.checkers)):
        points_dict[str(model.checkers[i])] = points[i]
    return points_dict


if __name__ == "__main__":
    log = xes_importer.apply("../../data/Sepsis Cases - Event Log_training.xes")
    score = run_all_mp_checkers_traces(log[0], '../../data/Sepsis_training_10_90_all.decl',
                                       '../../data/Sepsis_training_10_90_all.txt')
    print(score)
