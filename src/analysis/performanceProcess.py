
PATH = "../../cluster_data_old/output_performance/single_trace_testing_20_positional_cumulative.csv"

def main():
    perf_file = open(PATH, 'r')
    output_file = open(PATH.replace(".csv", "_processed.csv"), 'w')
    out_perf_file = open(PATH.replace(".csv", "sumup.csv"), 'w')
    perf_dict = dict()

    first = True
    for line in perf_file:
        if first:
            output_file.write(",".join(line.replace("\n", "").split(",") + ["delta_reward"]) + "\n")
            first = False
        else:
            parts = line.replace("\n","").split(",")
            if parts[7] != '':
                reward_difference = float(parts[7] if parts[7] != '' else 0) - float(parts[9])
            else:
                reward_difference = 0
            output_file.write(",".join(parts + [str(reward_difference)]) + "\n")
            if parts[2] not in perf_dict.keys():
                perf_dict[parts[2]] = {"count": 1, "sum_delta_reward": reward_difference}
            else:
                perf_dict[parts[2]]["count"] += 1
                perf_dict[parts[2]]["sum_delta_reward"] += reward_difference

    out_perf_file.write("perfix_len,count,avg_delta_reward\n")
    for k,d in perf_dict.items():
        out_perf_file.write(",".join([str(k),str(d["count"]), str(d["sum_delta_reward"]/d["count"])]) + "\n")
    out_perf_file.close()

    perf_file.close()
    output_file.close()


if __name__ == "__main__":
    main()