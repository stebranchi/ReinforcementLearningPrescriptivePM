import os

def main():
	PATH = os.path.join("..", "final_evaluation", "BPI", "performance", "performance_single_trace_testing_40_r005.csv")
	OUTPUT_PATH = os.path.join("..", "final_evaluation", "BPI", "performance_out", "prova")
	file = open(PATH, 'r')
	output_file = open(OUTPUT_PATH, 'w')

	PREFIX_LENGTH = 6
	first = True

	for line in file:
		if first:
			output_file.write(line)
			first = False
		elif int(line.split(",")[2]) == PREFIX_LENGTH:
			output_file.write(line)

	file.close()
	output_file.close()

if __name__ == "__main__":
	main()