import numpy as np

info = [3,4,3,2,0,1,5,3]
avg = np.zeros((2,4))
for episode_counter, num in enumerate(info,1):
    # print(avg, episode_counter, num)
    loc = (episode_counter-1)%4
    if loc == 0:
        if episode_counter-1 == 0:
            avg[0][0] = num
        else:
            avg[1][0] = (avg[0][-1]*(episode_counter-1) + num)/(episode_counter)
    else:
        if episode_counter-1 > 3:
            avg[1][loc] = (avg[1][loc-1]*(episode_counter-1) + num)/(episode_counter)
        else:
            avg[0][loc] = (avg[0][loc-1]*(episode_counter-1) + num)/(episode_counter)
print(avg)


def update_avg_value(current_list, epoch_num, episode_num, episode_sum, episodes_per_epoch, pulls_per_episode):
    i = epoch_num
    m = episode_num
    episode_counter = (m) + (i)*(episodes_per_epoch)
    episode_avg = episode_sum/pulls_per_episode

    if m == 0:
        if i == 0:
            current_list[i][m] = episode_avg
        else:
            current_list[i][m] = (
                current_list[i-1][-1]*(episode_counter) + episode_avg)/(episode_counter+1)
    else:
        current_list[i][m] = (
            current_list[i][m-1]*(episode_counter) + episode_avg)/(episode_counter+1)
    return current_list

avg = np.zeros((2,4))
pulls = 1
eps = 4
for i in range(2):
    for m in range(4):
        val = info[i*eps + m]
        avg = update_avg_value(avg, i, m, val, eps, pulls)
print(avg)

# def difference_of_weights(self, prior_vals):
#     layers = [self.i2h, self.h2h, self.a2c]
#     for layer_after, layer_before in zip(layers, prior_vals):
#         for (name, param_after), (name, param_before) in zip(layer_after.named_parameters(), layer_before.named_parameters()):
#             diff = torch.sub(param_after, param_before)
#             print(name, diff)


# a = [0.6808, 0.7325]
# b = [x/sum(a) for x in a]
# print(b)

# for _ in range(20):
#     val = np.random.random()
#     reward = int(val < 0.9)
#     print(val, reward)

# filename = 'C:\\Users\\joshc\\Google Drive\\CS Research\\Memory-Storage-Ritter\\dnd-lstm-sup-learning\\src\\emb_data.txt'
# # filename.replace('\\', '\\\\')
# print(filename)
# returns = []
# accuracy = []
# log_loss_val = []
# with open(filename, 'r') as f:
#     for line in f:
#         # print(line)
#         splits = line.split('|')
#         print(splits)
#         returns.append(float(splits[1][-5:-1]))
#         accuracy.append(float(line[-4:]))
#         loss = splits[2].split(',')
#         print(loss)
#         loc = loss[0].index('=')+2
#         loss_val = loss[0][loc:]
#         print(loss_val)
#         log_loss_val.append(float(loss_val))
#         # print(returns)
#         # print(accuracy)
#         break
# print(returns, accuracy, log_loss_val)
import torch
import random


def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.tensor(np_array).type(pth_dtype)

a = torch.zeros(3)
b = {}
b[a] = 1
print(b)
barcode_bag = set()
num_arms = 4
mapping = {}
num_barcodes = 6
barcode_size = 3
barcode_bag_list = []
while len(barcode_bag) <num_barcodes:
    prior = len(barcode_bag)
    barcode = np.random.randint(0, 2, (1, barcode_size))

    # Avoid cosine similarity bug with barcode of all 0's
    if np.sum(barcode) == 0:
        continue

    # barcode -> string starts out at '[[1 1 0]]', thus the reductions on the end
    barcode_string = np.array2string(barcode)[2:-2].replace(" ", "")
    barcode_bag.add(barcode_string)
    if len(barcode_bag) - prior == 1:
        barcode = to_pth(barcode)
        barcode_bag_list.append(barcode)


# Generate mapping of barcode to good arm
for barcode in barcode_bag_list:
    arm = random.randint(0, num_arms-1)
    mapping[barcode] = arm

# At least one barcode for every arm gets guaranteed
unique_guarantees = random.sample(barcode_bag_list, num_arms)
for arm, guarantee in enumerate(unique_guarantees):
    mapping[guarantee] = arm


print(mapping)


