import numpy as np

states_list = list()
states_dict = list()
csv = open("data/mdp/mdp_prob.tsv", 'r')
csv.readline()
for line in csv:
	values = line.strip().split("\t")
	if values[0] not in states_list:
		states_list.append(values[0])
	if values[2] not in states_list:
		states_list.append(values[2])
	tmp = dict()
	tmp["state"] = values[0]
	if "end" not in values[0]:
		tmp["action"] = values[1]
		tmp["next_state"] = values[2]
		tmp["prob"] = values[6]
		tmp["reward"] = float(values[11])
	states_dict.append(tmp)
csv.close()


'''==================================================
Initial set up
=================================================='''

#Hyperparameters
SMALL_ENOUGH = 0.005
GAMMA = 0.9
NOISE = 0.1

#Define all states
all_states=[]
for i in range(len(states_list)):
	all_states.append(i)

#Define rewards for all states
rewards = {}
for i in all_states:
	for item in states_dict:
		if item["state"] == states_list[i]:
			rewards[i] = item["reward"]

	if i not in rewards.keys():
		rewards[i] = 0
#if 'END' in states_list[i]:
#	rewards[i] = 10
#else:
#	rewards[i] = 0

#Dictionnary of possible actions. We have two "end" states (1,2 and 2,2)
actions = dict()
for element in states_dict:
	if element["state"] in actions.keys() and "action" in element.keys():
		actions[element["state"]].append(element["action"])
	elif "action" in element.keys():
		actions[element["state"]] = [element["action"]]

#Define an initial policy
policy={}
for s in actions.keys():
	policy[s] = np.random.choice(actions[s])

#Define initial value function
V={}
for s in all_states:
	V[s] = rewards[s]

'''==================================================
Value Iteration
=================================================='''

iteration = 0
while True:
	biggest_change = 0
	for s in all_states:
		if states_list[s] in policy:

			for a in actions[states_list[s]]:
				states = list()
				probs = list()
				for item in states_dict:
					if item["state"] == states_list[s] and item["action"] == a:
						states.append(item["next_state"])
						probs.append(item["prob"])
				p = np.array(probs).astype(np.float)
				p /= p.sum()
				nxt = states_list.index(np.random.choice(states, p=p))


	#See if the loop should stop now
	if biggest_change < SMALL_ENOUGH:
		out = open("MDP_r2_policy.txt", 'w')
		out.write(str(policy))
		out.close()
		print(iteration)
		print(policy)
		break
	iteration += 1
