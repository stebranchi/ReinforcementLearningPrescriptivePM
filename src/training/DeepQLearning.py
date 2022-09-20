from sklearn.externals import joblib
import numpy as np
import tensorflow as tf
import os
import time
import math

from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from tensorflow import keras
from tensorflow.keras import layers
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log import log as lg
from datetime import datetime, timedelta
from pm4py.objects.log.importer.xes import importer as xes_importer



def main():
	# Configuration parameters for the whole setup
	seed = 42
	gamma = 0.99  # Discount factor for past rewards
	max_steps_per_episode = 10000
	env = myEnv()
	env.customInit("data/nets/petriNet.pnml", "data/models/job_60-split_4-predictive_model-prediction-v0.sav")
	#env = gym.make("CartPole-v0")  # Create the environment
	#env.seed(seed)
	eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

	"""
	## Implement Actor Critic network
	This network learns two functions:
	1. Actor: This takes as input the state of our environment and returns a
	probability value for each action in its action space.
	2. Critic: This takes as input the state of our environment and returns
	an estimate of total rewards in the future.
	In our implementation, they share the initial layer.
	"""

	num_inputs = 8
	num_actions = 9
	num_hidden = 128

	inputs = layers.Input(shape=(num_inputs,))
	common = layers.Dense(num_hidden, activation="relu")(inputs)
	action = layers.Dense(num_actions, activation="softmax")(common)
	critic = layers.Dense(1)(common)

	model = keras.Model(inputs=inputs, outputs=[action, critic])

	"""
	## Train
	"""

	optimizer = keras.optimizers.Adam(learning_rate=0.01)
	huber_loss = keras.losses.Huber()
	action_probs_history = []
	critic_value_history = []
	rewards_history = []
	best_ep_reward = 0
	running_reward = 0
	episode_count = 0

	while True:  # Run until solved
		state = env.reset()
		episode_reward = 0
		with tf.GradientTape() as tape:
			for timestep in range(1, max_steps_per_episode):
				# env.render(); Adding this line would show the attempts
				# of the agent in a pop up window.

				state = tf.convert_to_tensor(state)
				state = tf.expand_dims(state, 0)

				# Predict action probabilities and estimated future rewards
				# from environment state
				action_probs, critic_value = model(state)
				critic_value_history.append(critic_value[0, 0])

				# Sample action from action probability distribution
				action = np.random.choice(num_actions, p=np.squeeze(action_probs))
				action_probs_history.append(tf.math.log(action_probs[0, action]))

				# Apply the sampled action in our environment
				state, reward, done = env.step(action)
				rewards_history.append(reward)
				episode_reward += reward

				if done:
					#print(episode_reward)
					break

			# Update running reward to check condition for solving
			running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

			# Calculate expected value from rewards
			# - At each timestep what was the total reward received after that timestep
			# - Rewards in the past are discounted by multiplying them with gamma
			# - These are the labels for our critic
			returns = []
			discounted_sum = 0
			for r in rewards_history[::-1]:
				discounted_sum = r + gamma * discounted_sum
				returns.insert(0, discounted_sum)

			# Normalize
			returns = np.array(returns)
			returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
			returns = returns.tolist()

			# Calculating loss values to update our network
			history = zip(action_probs_history, critic_value_history, returns)
			actor_losses = []
			critic_losses = []
			for log_prob, value, ret in history:
				# At this point in history, the critic estimated that we would get a
				# total reward = `value` in the future. We took an action with log probability
				# of `log_prob` and ended up recieving a total reward = `ret`.
				# The actor must be updated so that it predicts an action that leads to
				# high rewards (compared to critic's estimate) with high probability.
				diff = ret - value
				actor_losses.append(-log_prob * diff)  # actor loss

				# The critic must be updated so that it predicts a better estimate of
				# the future rewards.
				critic_losses.append(
					huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
				)

			# Backpropagation
			loss_value = sum(actor_losses) + sum(critic_losses)
			grads = tape.gradient(loss_value, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))

			# Clear the loss and reward history
			action_probs_history.clear()
			critic_value_history.clear()
			rewards_history.clear()
		# Log details
		episode_count += 1
		if episode_count % 10 == 0:
			template = "running_reward reward: {:.2f} at episode {}"
			print(template.format(running_reward, episode_count))
			#print(running_reward)
		if episode_count > 1000:
			print("The problem couldn't be solved in 1000 episodes")
			break
		if running_reward > 180:  # Condition to consider the task solved
			print("Solved at episode {} with episode reward {}!".format(episode_count, running_reward))
			break


def evaluate(model):
	return 0


class myEnv():
	state = []
	complete_trace = []
	initial_marking = None
	final_marking = None
	model = None
	net = None
	log = None
	index_action = 0
	possible_actions = ["A", "B", "C", "D", "E", "F", "G", "H"]

	def step(self, action):
		# time.sleep(0.5)
		# print("action:" + str(action))
		done = False
		reward = 1
		if int(action) == 8:
			done = True
			# print(self.state)
			checker = self.checkTrace()
			if checker['trace_is_fit']:
				reward = 10
		elif self.index_action > 7:
			self.shift()
			self.state[-1] = action
			self.complete_trace[self.index_action] = action
			self.index_action += 1
		else:
			self.state[self.index_action] = action
			self.complete_trace[self.index_action] = action
			self.index_action += 1
			checker = self.checkTrace()
			# IF U WANT TO USE REMAINING TIME
			# reward = self.checkRemainingTime()/2
			#IF U WANT TO USE LABEL
			reward = self.checkLabel()
			reward += len(checker['activated_transitions']) * 3 - len(checker['transitions_with_problems'])
		return self.state, reward, done

	def reset(self):
		self.state = np.zeros((8,))
		self.complete_trace = []
		for i in range(8):
			self.state[i] = -1
		self.index_action = 0
		return self.state

	def checkRemainingTime(self):
		reward = 0.0
		c = 0
		for trace in self.log:
			satisfied = True
			for index in range(self.index_action):
				if len(trace) > index+1:
					if trace[index+1]["concept:name"] != self.possible_actions[int(self.state[index])]:
						satisfied = False
			if satisfied:
				c += 1
				time = trace[-1]["time:timestamp"] - trace[self.index_action-1]["time:timestamp"]
				# print("time" + str(time.total_seconds()))
				# TODO: change fixed value with the time of the longest trace in the xes
				reward += float(1000000000.0/time.total_seconds())
				# print(reward)
		if c == 0:
			return 0
		else:
			return reward/c

	def customInit(self, net_path, model_path):
		self.model = joblib.load(model_path)
		self.log = xes_importer.apply('data/logs/simple_example2_no_lifecycle.xes')
		net, initial_marking, final_marking = pnml_importer.apply(net_path)
		self.net = net
		self.initial_marking = initial_marking
		self.final_marking = final_marking

	def checkTrace(self):
		log = self.createLog()
		replayed_traces = token_replay.apply(log, self.net, self.initial_marking, self.final_marking)
		return replayed_traces[0]

	def checkLabel(self):
		reward = 10
		trace_arr = np.array(self.complete_trace)
		result = self.model[0].predict(trace_arr.reshape(1, -1))
		reward *= result[0]
		return reward

	def createLog(self):
		log= lg.EventLog()
		trace = lg.Trace()
		trace.attributes["concept:name"] = 1
		c = 1
		end = False
		event = lg.Event()
		event["concept:name"] = "START"
		event["time:timestamp"] = (datetime.now()).timestamp()
		trace.append(event)
		for action in range(self.index_action):
			event = lg.Event()
			if not end:
				if self.complete_trace[action] == 8:
					event["concept:name"] = "END"
					end = True
				else:
					event["concept:name"] = self.possible_actions[int(self.state[action])]
				event["time:timestamp"] = (datetime.now() + timedelta(hours=c)).timestamp()
				c += 1
			trace.append(event)
		log.append(trace)
		return log

	def shift (self):
		for i in range(len(self.state)):
			if i > 0:
				self.state[i-1] = self.state[i]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	main()

