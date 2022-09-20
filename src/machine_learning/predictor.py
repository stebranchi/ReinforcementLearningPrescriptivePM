import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier

import pydotplus
from IPython.display import Image

from src.models import *
from src.machine_learning.encoder import *

from pm4py.objects.log.importer.xes import factory as xes_import_factory
log = xes_import_factory.apply(IN_LOG_PATH)


def create_decision_tree(log, speed_threshold, probability_threshold, activation_rules, corellation_rules):
    feature_names, encoded_data = encode_trace(log, probability_threshold, activation_rules, corellation_rules)
    categories = [TraceSpeed.SLOW.value, TraceSpeed.FAST.value]

    X = pd.DataFrame(encoded_data, columns=feature_names)
    y = pd.Categorical(get_log_speed(log, speed_threshold), categories=categories)

    dtc = DecisionTreeClassifier(random_state=0)
    dtc.fit(X, y)

    dot_data = tree.export_graphviz(dtc, out_file=None, feature_names=feature_names, class_names=["Slow", "Fast"],
                                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())
    graph.write_pdf(OUT_PDF_DECISION_TREE_PATH)

    tree_rules = export_text(dtc, feature_names=feature_names, show_weights=True)
    print(tree_rules)
    return dtc, feature_names


def generate_paths(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    features = [feature_names[i] for i in tree.tree_.feature]
    leaf_ids = np.argwhere(left == -1)[:, 0]
    leaf_ids_fast = filter(lambda leaf_id: tree.tree_.value[leaf_id][0][0] < tree.tree_.value[leaf_id][0][1], leaf_ids)

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            state = TraceState.VIOLATED
        else:
            parent = np.where(right == child)[0].item()
            state = TraceState.SATISFIED

        lineage.append((features[parent], state))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    paths = []
    for leaf_id in leaf_ids_fast:
        path = PathModel()
        rules = []
        for node in recurse(left, right, leaf_id):
            rules.append(node)
        path.rules = rules
        path.impurity = tree.tree_.impurity[leaf_id]
        paths.append(path)
    return paths


def parse_method(method):
    method_name = method.split("[")[0]
    rest = method.split("[")[1][:-1]
    if "," in rest:
        method_params = rest.split(",")
    else:
        method_params = [rest]
    return method_name, method_params


def call_method(trace, done, method_name, method_params, activation_rules, corellation_rules):
    result = None
    if method_name == RESPONDED_EXISTENCE:
        result = mp_responded_existence_checker(trace, done, method_params[0], method_params[1], activation_rules, corellation_rules)
    elif method_name == RESPONSE:
        result = mp_response_checker(trace, done, method_params[0], method_params[1], activation_rules, corellation_rules)
    elif method_name == ALTERNATE_RESPONSE:
        result = mp_alternate_response_checker(trace, done, method_params[0], method_params[1], activation_rules, corellation_rules)
    elif method_name == CHAIN_RESPONSE:
        result = mp_chain_response_checker(trace, done, method_params[0], method_params[1], activation_rules, corellation_rules)
    elif method_name == PRECEDENCE:
        result = mp_precedence_checker(trace, done, method_params[0], method_params[1], activation_rules, corellation_rules)
    elif method_name == ALTERNATE_PRECEDENCE:
        result = mp_alternate_precedence_checker(trace, done, method_params[0], method_params[1], activation_rules, corellation_rules)
    elif method_name == CHAIN_PRECEDENCE:
        result = mp_chain_precedence_checker(trace, done, method_params[0], method_params[1], activation_rules, corellation_rules)
    return result


def make_decision(trace, path, activation_rules, corellation_rules):
    recommendation = ""
    for rule in path.rules:
        method, state = rule
        method_name, method_params = parse_method(method)
        result = call_method(trace, False, method_name, method_params, activation_rules, corellation_rules)
        print("method:", method, "tree:", state, "trace:", result.state)
        if state == TraceState.SATISFIED:
            if result.state == TraceState.VIOLATED:
                recommendation = "Contradiction"
                break
            elif result.state == TraceState.SATISFIED:
                pass
            elif result.state == TraceState.POSSIBLY_VIOLATED:
                recommendation += method + " should be SATISFIED. "
            elif result.state == TraceState.POSSIBLY_SATISFIED:
                recommendation += method + " should not be VIOLATED. "
        elif state == TraceState.VIOLATED:
            if result.state == TraceState.VIOLATED:
                pass
            elif result.state == TraceState.SATISFIED:
                recommendation = "Contradiction"
                break
            elif result.state == TraceState.POSSIBLY_VIOLATED:
                recommendation += method + " should not be SATISFIED. "
            elif result.state == TraceState.POSSIBLY_SATISFIED:
                recommendation += method + " should be VIOLATED. "
    return recommendation


def make_decisions(trace, log, speed_threshold, probability_threshold, activation_rules, correlation_rules):
    tree, feature_names = create_decision_tree(log, speed_threshold, probability_threshold, activation_rules, correlation_rules)
    paths = generate_paths(tree, feature_names)
    for path in paths:
        print("path: ", path.rules)
        print("impurity: ", path.impurity)
        recommendation = make_decision(trace, path, activation_rules, correlation_rules)
        print(recommendation)
        print("==================================================")


make_decisions(log[0], log, 12000, 0, "True", "True")
