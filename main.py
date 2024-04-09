import torch
import os
from MLP import MultiLayerPerceptron as MLP
from ActorCritic import TDActorCritic
from train_utils import to_tensor, prepare_training_inputs
from Archery_env.Archery import ArcheryEnv
from memory import BatchMemory


import matplotlib.pyplot as plt
from scipy.io import savemat
from collections import deque
from param import Hyper_Param
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'

# Hyperparameters
lr = Hyper_Param['learning_rate']
batch_size = Hyper_Param['batch_size']
gamma = Hyper_Param['discount_factor']
total_eps = Hyper_Param['num_episode']
print_every = Hyper_Param['print_every']
window_size = Hyper_Param['window_size']
step_max = Hyper_Param['step_max']
beta = Hyper_Param['beta']
beta_min = Hyper_Param['beta_min']
beta_decay_rate = Hyper_Param['beta_decay_rate']

# List storing the results
score_avg = deque(maxlen=window_size)
cum_score_list = []
score_avg_value = []
epi = []

# Create Environment
env = ArcheryEnv()
s_dim = env.state_dim
a_num = env.action_space.n


policy_net = MLP(s_dim, a_num, num_neurons=Hyper_Param['num_neurons'])
value_net = MLP(s_dim, 1, num_neurons=Hyper_Param['num_neurons'])

agent = TDActorCritic(policy_net, value_net, beta, batch_size, gamma, lr)
memory = BatchMemory(batch_size)

# Episode start
for n_epi in range(total_eps):
    s = env.reset()
    agent.beta = max(beta_min, beta*beta_decay_rate)
    epi.append(n_epi)
    while True:
        s = to_tensor(s, size=(3,))
        a = agent.get_action(s).view(-1, 1)
        ns, r, done, info = env.step(a.item())
        experience = (s.view(1, 3),
                      a,
                      torch.tensor(r).view(1, 1),
                      torch.tensor(ns).view(1, 3),
                      torch.tensor(done).view(1, 1))
        memory.push(experience)
        env.state = ns
        s = env.state

        if done:
            break

    sampled_exps = memory.sample(batch_size)
    sampled_exps = prepare_training_inputs(sampled_exps)
    agent.update(*sampled_exps)
    memory.reset()

    cum_score = env.cum_score/step_max
    score_avg.append(cum_score)
    cum_score_list.append(cum_score)

    if len(score_avg) == window_size:
        score_avg_value.append(sum(score_avg) / window_size)
    else:
        score_avg_value.append(sum(score_avg) / len(score_avg))

    if n_epi % print_every == 0:
        msg = (n_epi, cum_score)
        print("Episode : {:4.0f} | Cumulative score : {:.2f}".format(*msg))
        plt.xlim(0, total_eps)
        plt.ylim(0, 10)
        plt.plot(epi, cum_score_list, color='black')
        plt.plot(epi, score_avg_value, color='red')
        plt.xlabel('Episode', labelpad=5)
        plt.ylabel('Average score', labelpad=5)
        plt.grid(True)
        plt.pause(0.0001)
        plt.close()


# Base directory path creation
base_directory = os.path.join(Hyper_Param['today'])

# Subdirectory index calculation
if not os.path.exists(base_directory):
    os.makedirs(base_directory)
    index = 1
else:
    existing_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    indices = [int(d) for d in existing_dirs if d.isdigit()]
    index = max(indices) + 1 if indices else 1

# Subdirectory creation
sub_directory = os.path.join(base_directory, str(index))
os.makedirs(sub_directory)

# Store plt in Subdirectory
plt.xlim(0, total_eps)
plt.ylim(0, 10)
plt.plot(epi, cum_score_list, color='black')
plt.plot(epi, score_avg_value, color='red')
# plt.plot(epi, optimal_score_avg_value, color='blue')
# plt.plot(epi, cum_rand_score_list, color='blue')
# plt.plot(epi, cum_optimal_score_list, '--g')
plt.xlabel('Episode', labelpad=5)
plt.ylabel('Average score', labelpad=5)
plt.grid(True)
plt.savefig(os.path.join(sub_directory, f"plot_{index}.png"))

# Store Hyperparameters in txt file
with open(os.path.join(sub_directory, 'Hyper_Param.txt'), 'w') as file:
    for key, value in Hyper_Param.items():
        file.write(f"{key}: {value}\n")

# Store score data (matlab data file)
savemat(os.path.join(sub_directory, 'data.mat'),{'sim_res': cum_score_list})
# savemat(os.path.join(sub_directory, 'data.mat'),{'sim_res': cum_score_list,'sim_optimal': optimal_score_avg_value})
# savemat(os.path.join(sub_directory, 'data.mat'), {'sim_res': cum_score_list,'sim_rand_res': cum_rand_score_list,
#                                                   'sim_optimal_res': cum_optimal_score_list})
