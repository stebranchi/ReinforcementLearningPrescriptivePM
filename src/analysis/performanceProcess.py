
PATH = "../../cluster_data/normal_encoding/output_performances/BPI2012_100_notscaled_step_test2.csv"

def main():
    perf_file = open(PATH, 'r')
    output_path = PATH.replace("output_performances", "performance_results")
    output_file = open(output_path.replace(".csv", "_processed.csv"), 'w')
    out_perf_file = open(output_path.replace(".csv", "sumup.csv"), 'w')
    perf_dict = dict()

    first = True
    for line in perf_file:
        if first:
            output_file.write(",".join(line.replace("\n", "").split(",") + ["delta_reward", "delta_%O_ACCEPTED"]) + "\n")
            first = False
        else:
            parts = line.replace("\n","").split(",")
            if parts[7] != '':
                reward_difference = float(parts[7] if parts[7] != '' else 0) - float(parts[9])
            else:
                reward_difference = 0
            if parts[10] != '':
                o_accepted_diff = float(parts[10]) - float(parts[12])
            else:
                o_accepted_diff = 0
            output_file.write(",".join(parts + [str(reward_difference), str(o_accepted_diff)]) + "\n")
            if parts[2] not in perf_dict.keys():
                perf_dict[parts[2]] = {"count": 1, "sum_delta_reward": reward_difference, "sum_delta_o_accepted": o_accepted_diff}
            else:
                perf_dict[parts[2]]["count"] += 1
                perf_dict[parts[2]]["sum_delta_reward"] += reward_difference
                perf_dict[parts[2]]["sum_delta_o_accepted"] += o_accepted_diff

    out_perf_file.write("prefix_len,count,avg_delta_reward,avg_delta_%O_ACCEPTED\n")
    for k,d in perf_dict.items():
        out_perf_file.write(",".join([str(k),str(d["count"]), str(d["sum_delta_reward"]/d["count"]), str(d["sum_delta_o_accepted"]/d["count"])]) + "\n")
    out_perf_file.close()

    perf_file.close()
    output_file.close()


if __name__ == "__main__":
    main()