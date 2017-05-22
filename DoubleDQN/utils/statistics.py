import json
import matplotlib.pyplot as plt
import argparse

# Parameter settings
parser = argparse.ArgumentParser(description='Gain statistics.')

parser.add_argument('--episode-num', type=int, default='30', metavar='N',
        help='the number of episodes in one evaluation (default:30)')
parser.add_argument('--eval-num', type=int, default=100, metavar='N',
        help='the number of the evaluations occured during training (default:100)')
parser.add_argument('--eval-freq', type=int, default=20000, metavar='N',
        help='the frquency of the evaluation (default:20000)')

args = parser.parse_args()


def calculate_metrics_in_evaluation(idx, episode_num):
	# create file names
	file_name_base = "files/rewards/reward" + str(idx) + "_"
	
	metrics = {}
	average_reward = 0.0
	minimum_reward = 0.0
	maximum_reward = 0.0
	
	file_name = file_name_base + str(1) + ".json"
	with open(file_name, 'r') as f:
		x = json.load(f)
		rw = x[-1]
		average_reward += rw
		maximum_reward = rw
		minimum_reward = rw
	
	for eps in range(2, episode_num + 1):
		file_name = file_name_base + str(eps) + ".json"
		with open(file_name, 'r') as f:
			x = json.load(f)
			rw = x[-1]
			average_reward += rw
			if maximum_reward < rw:
				maximum_reward = rw
			if minimum_reward > rw:
				minimum_reward = rw
				
	metrics['avg'] = average_reward / float(episode_num)
	metrics['max'] = maximum_reward
	metrics['min'] = minimum_reward
	
	return metrics
	
def plot_curves(evaluation_frequency, evaluation_num, episode_num):
	
	# gather the matric data
	steps = [0] * evaluation_num
	avgs = [0] * evaluation_num
	maxs = [0] * evaluation_num
	mins = [0] * evaluation_num
	for i in range(0, evaluation_num):
		steps[i] = evaluation_frequency * (i+1)
		metrics = calculate_metrics_in_evaluation(i+1, episode_num)
		avgs[i] = metrics['avg']
		maxs[i] = metrics['max']
		mins[i] = metrics['min']
		
	# plot the curves
	plt.figure(1)
	plt.plot(steps, avgs)
	plt.xlabel('Number of steps')
	plt.ylabel('Average returns')
	plt.show()
	
	plt.figure(2)
	plt.plot(steps, maxs)
	plt.xlabel('Number of steps')
	plt.ylabel('Maximum returns')
	plt.show()
	
	plt.figure(3)
	plt.plot(steps, mins)
	plt.xlabel('Number of steps')
	plt.ylabel('Minimum returns')
	plt.show()
	
# CALLING THE PLOT FUNCTION

plot_curves(args.eval_freq, args.eval_num, args.episode_num)