import numpy as np
import sys
import math

LOG = "data/mdp/SB_simple_model_MDP_r2.csv"

states_list = list()
states_dict = list()
csv = open(LOG, 'r')
filename = LOG.split("/")[-1].split(".")[0]
csv.readline()
for line in csv:
	values = line.strip().split(",")
	if values[0] not in states_list:
		states_list.append(values[0])
	if values[2] not in states_list:
		states_list.append(values[2])
	tmp = dict()
	tmp["state"] = values[0]
	if "end" not in values[0]:
		tmp["action"] = values[1]
		tmp["next_state"] = values[2]
		tmp["prob"] = values[3]
		tmp["reward"] = float(values[4])
	states_dict.append(tmp)
csv.close()


'''==================================================
Initial set up
=================================================='''

#Hyperparameters
SMALL_ENOUGH = 0.005
GAMMA = 0.9
NOISE = 0.1
alpha = 0.1

#Define all states
all_states = []
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

#Dictionnary of possible actions. We have two "end" states (1,2 and 2,2)
actions = dict()
for element in states_dict:
	if element["state"] in actions.keys() and "action" in element.keys():
		actions[element["state"]].add(element["action"])
	elif "action" in element.keys():
		actions[element["state"]] = set([element["action"]])

for key in actions.keys():
	actions[key] = list(actions[key])

#Define an initial policy
policy = {}
for s in actions.keys():
	policy[s] = np.random.choice(list(actions[s]))

#Define initial value function
V = {}
for s in all_states:
	V[states_list[s]] = -sys.maxsize

'''==================================================
Value Iteration
=================================================='''

iteration = 0
while iteration < 1000:
	biggest_change = 0
	for s in all_states:
		if states_list[s] in policy:

			old_v = V[states_list[s]]
			new_v = V[states_list[s]]
			for a in actions[states_list[s]]:
				states = list()
				probs = list()
				for item in states_dict:
					if item["state"] == states_list[s] and item["action"] == a:
							states.append(item["next_state"])
							probs.append(item["prob"])
							reward = item["reward"]
				p = np.array(probs).astype(np.float)
				p /= p.sum()
				nxt = states_list.index(np.random.choice(states, p=p))

				#Choose a new random action to do (transition probability)
				if len([i for i in actions[states_list[s]] if i != a]) > 0:
					random_1 = np.random.choice([i for i in actions[states_list[s]] if i != a])
					states = list()
					probs = list()
					for item in states_dict:
						if item["state"] == states_list[s] and 'action' in item.keys() and item["action"] == random_1:
							states.append(item["next_state"])
							probs.append(item["prob"])
							reward_noise = item["reward"]
					p = np.array(probs).astype(np.float)
					p /= p.sum()
					act = states_list.index(np.random.choice(states, p=p))
					#v = (1-alpha) * old_v + alpha * (GAMMA * ((1-NOISE) * (reward + V[states_list[nxt]]) + (NOISE * (reward_noise + V[states_list[act]]))))
					v = (1-alpha) * old_v + alpha * (GAMMA * (reward + V[states_list[nxt]]))
				else:
					v = (1-alpha) * old_v + alpha * (reward + (GAMMA * V[states_list[nxt]]))
				"""if random_1 == 'U':
					act = [s[0]-1, s[1]]
				if random_1 == 'D':
					act = [s[0]+1, s[1]]
				if random_1 == 'L':
					act = [s[0], s[1]-1]
				if random_1 == 'R':
					act = [s[0], s[1]+1]"""
				#Calculate the value
				#nxt = tuple(nxt)
				#act = tuple(act)
				if v > new_v: #Is this the best action so far? If so, keep it
					new_v = v
					policy[states_list[s]] = a
			#Save the best of all actions for the state
			V[states_list[s]] = new_v
			biggest_change = max(biggest_change, np.abs(old_v - V[states_list[s]]))

	#See if the loop should stop now
	if biggest_change < SMALL_ENOUGH:
		print(V)
		#out = open("MDP_r2_policy.txt", 'w')
		#out.write(str(policy))
		#out.close()
		print(iteration)
		print(policy)
		break
	iteration += 1

print(V)
out = open("results/" + filename + ".csv", 'w')
out.write("s, V(s), a(s)\n")
for key, item in policy.items():
	out.write(','.join([key, str(V[key]), item])+ '\n')
out.close()
