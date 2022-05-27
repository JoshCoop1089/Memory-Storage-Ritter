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

.4, .8, .2, .6