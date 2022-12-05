import pandas as pd
import copy
import pm4py
import datetime
from pm4py.objects.log.importer.xes import importer as xes_importer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import kosaraju
from pymining import seqmining
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.algo.filtering.log.variants import variants_filter

environment_action = ["O_ACCEPTED", "A_CANCELLED", "O_SENT_BACK"]
min_support = 3

def main(path_in):
    log = xes_importer.apply(path_in)
    matrix = []
    for trace in log:
        matrix.append([e["concept:name"] for e in trace])
    a = TransactionEncoder()
    a_data = a.fit(matrix).transform(matrix)
    df = pd.DataFrame(a_data, columns=a.columns_)
    df = df.replace(False, 0)
    df_association_rules = apriori(df, min_support=0.6, use_colnames=True, verbose=1)
    df_association_rules = df_association_rules.sort_values('support')
    print(df_association_rules)


def seqDiscovery(log):
    seqs = [[x["concept:name"] for x in trace] for trace in log[:10]]
    report = seqmining.freq_seq_enum(seqs, 10)
    print(report)


def removeComponents(log):
    #filtered_log = variants_filter.filter_variants_by_coverage_percentage(log, min_coverage_percentage=0.0001)
    filtered_log = log
    G = kosaraju.DiGraph()
    events_set = set()
    vertex_dict = dict()
    connections_set = set()
    connections_count = dict()
    for trace in filtered_log:
        last_event = None
        for e in trace:
            events_set.add(e["concept:name"])
            if last_event:
                key = (last_event, e["concept:name"])
                if key in connections_count.keys():
                    connections_count[key] += 1
                else:
                    connections_count[key] = 1
                connections_set.add((last_event, e["concept:name"]))
            last_event = e["concept:name"]

    for e in events_set:
        vertex_dict[e] = kosaraju.Vertex(e)

    G.add_vertices([v for e, v in vertex_dict.items()])

    #for (e1, e2) in connections_set:
    #    G.add_edge(vertex_dict[e1], vertex_dict[e2])

    for k, c in connections_count.items():
        if c > min_support:
            G.add_edge(vertex_dict[k[0]], vertex_dict[k[1]])

    y = kosaraju.kosaraju(G)

    for j in range(len(y)):
        if y[j] != []:
            print("Component:", j + 1, " ", end=" ")
            for v in y[j]:
                print(v.value, end=" ")
            print()


    print(len(filtered_log) / len(log))


def orderEventsOnTimestamps(log_path, remove_micro=False):
    log = pm4py.read_xes(log_path)
    output_log = EventLog()
    for name, value in log.attributes.items():
        output_log.attributes[name] = value

    for trace in log:
        trace_timestamps_set = set()
        events_by_timestamp = dict()
        for event in trace[1:-1]:
            ev = copy.deepcopy(event)
            if remove_micro:
                ev["time:timestamp"] = ev["time:timestamp"].replace(microsecond=0)
            timestamp = ev["time:timestamp"]
            # timestamp = event["time:timestamp"].strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            trace_timestamps_set.add(timestamp)
            if event["concept:name"] in environment_action:
                if timestamp in events_by_timestamp.keys():
                    events_by_timestamp[timestamp]["env"].append(ev)
                else:
                    events_by_timestamp[timestamp] = {"agent": [], "env": [ev]}
            else:
                if timestamp in events_by_timestamp.keys():
                    events_by_timestamp[timestamp]["agent"].append(ev)
                else:
                    events_by_timestamp[timestamp] = {"agent": [ev], "env": []}
        new_trace = Trace()
        for name, value in trace.attributes.items():
            new_trace.attributes[name] = value
        new_trace.append(trace[0])
        if remove_micro:
            new_trace[0]["time:timestamp"] = new_trace[0]["time:timestamp"].replace(microsecond=0)
        for timestamp in sorted(list(trace_timestamps_set)):
            [new_trace.append(x) for x in sorted(events_by_timestamp[timestamp]["env"], key=lambda x: x["concept:name"])]
            [new_trace.append(x) for x in sorted(events_by_timestamp[timestamp]["agent"], key=lambda x: x["concept:name"])]
        new_trace.append(trace[-1])
        if remove_micro:
            new_trace[-1]["time:timestamp"] = new_trace[-1]["time:timestamp"].replace(microsecond=0)
        output_log.append(new_trace)

    if remove_micro:
        pm4py.write_xes(output_log, log_path.replace(".xes", "_ordered_nomicro.xes"))
    else:
        pm4py.write_xes(output_log, log_path.replace(".xes", "_ordered_micro.xes"))


def extrateSameTimeEvents(log_path, micro=False):
    log = pm4py.read_xes(log_path)

    same_timestamps_events_set_counter = dict()
    for trace in log:
        events_by_timestamp = dict()
        for event in trace[1:-1]:
            if micro:
                timestamp = event["time:timestamp"].replace(microsecond=0)
            else:
                timestamp = event["time:timestamp"]
            if timestamp in events_by_timestamp.keys():
                events_by_timestamp[timestamp].add(event["concept:name"])
            else:
                events_by_timestamp[timestamp] = {event["concept:name"]}
        for k, v in events_by_timestamp.items():
            if len(list(v)) > 1:
                if tuple(v) in same_timestamps_events_set_counter.keys():
                    same_timestamps_events_set_counter[tuple(v)] += 1
                else:
                    same_timestamps_events_set_counter[tuple(v)] = 1

    return same_timestamps_events_set_counter





if __name__ == "__main__":
    #main("../../cluster_data/output_logs/BPI2012_log_eng_positional_cumulative_squashed_training_80.xes")
    #removeComponents(pm4py.read_xes("../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations.xes"))
    #seqDiscovery(pm4py.read_xes("../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations.xes"))
    orderEventsOnTimestamps("../../data/logs/BPI_2012/BPI_2012_log_eng.xes", True)
    #same_timestamps_events_set_counter = extrateSameTimeEvents("../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations.xes")
    #same_timestamps_micro_events_set_counter = extrateSameTimeEvents("../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations.xes", True)
    #print(set(same_timestamps_events_set_counter).intersection(set(same_timestamps_micro_events_set_counter)))
