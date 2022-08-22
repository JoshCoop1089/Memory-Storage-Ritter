# a = ['original']
# b = ['update', 'update_avg', 'update_ind_avg']
# mem_update_boolean = any(x in max(a,b,key=len) for x in min(b,a,key=len))
# print(mem_update_boolean)


# trial_hidden_states = [(0, 1), (1, 3), (2, 1), (3, 2), (4, -1)]

# # Find any multi pulls on same memory
# unique_id = set([int(x[1]) for x in trial_hidden_states if x[1] != -1])
# unique_id_dict = {key: [] for key in unique_id}
# print(unique_id_dict)
# for state, mem_id in trial_hidden_states:
#     if mem_id != -1:
#         unique_id_dict[mem_id].append(state)

# print(unique_id_dict)
# new_keys = []
# no_matches = [x[0] for x in trial_hidden_states if x[1] == -1]
# for x in no_matches:
#     new_keys.append([x])
# print(new_keys)


# from sklearn.manifold import TSNE
# import numpy as np
# import matplotlib as plt
# """
# tsne = TSNE(n_components=2).fit_transform(features)
# # This is it — the result named tsne is the 2-dimensional projection of the 2048-dimensional features. 
# # n_components=2 means that we reduce the dimensions to two. Here we use the default values of all 
# # the other hyperparameters of t-SNE used in sklearn.

# # Okay, we got the t-SNE — now let’s visualize the results on a plot. 
# # First, we’ll normalize the points so they are in [0; 1] range.
# # scale and move the coordinates so they fit [0; 1] range
# def scale_to_01_range(x):
#     # compute the distribution range
#     value_range = (np.max(x) - np.min(x))
#     # move the distribution so that it starts from zero
#     # by extracting the minimal value from all its values
#     starts_from_zero = x - np.min(x)
#     # make the distribution fit [0; 1] by dividing by its range
#     return starts_from_zero / value_range
# # extract x and y coordinates representing the positions of the images on T-SNE plot
# tx = tsne[:, 0]
# ty = tsne[:, 1]
# tx = scale_to_01_range(tx)
# ty = scale_to_01_range(ty)
# # Now let’s plot the 2D points, each in a color corresponding to its class label.
# # initialize a matplotlib plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # # for every class, we'll add a scatter plot separately
# # for label in colors_per_class:
# #     # find the samples of the current class in the data
# #     indices = [i for i, l in enumerate(labels) if l == label]
# #     # extract the coordinates of the points of this class only
# #     current_tx = np.take(tx, indices)
# #     current_ty = np.take(ty, indices)
# #     # convert the class color to matplotlib format
# #     color = np.array(colors_per_class[label], dtype=np.float) / 255
# #     # add a scatter plot with the corresponding color and label

# #     ax.scatter(current_tx, current_ty, c=color, label=label)

# # # build a legend using the labels we set previously

# # ax.legend(loc='best')
# # finally, show the plot
# plt.show()
# """
# a = [{"target": 0.1890480000000001, "params": {'dim_hidden_a2c': 8.492976777701108,'dim_hidden_lstm': 7.179425597794003, "entropy_error_coef": 0.06526811936609955, "lstm_learning_rate": -3.1330933661544993, "value_error_coef": 0.40681007049253626}},
#     {"target": 0.1869, "params": {'dim_hidden_a2c': 7.970393537914677,'dim_hidden_lstm': 7.595443984589479, "entropy_error_coef": 2.859, "lstm_learning_rate": -2.519418213170831, "value_error_coef": 0.1468}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.0, "delta": 0.0}},
#     {"target": 0.187, "params": {'dim_hidden_a2c': 6.719731056749852,'dim_hidden_lstm': 6.203984165855989, "entropy_error_coef": 0.08639, "lstm_learning_rate": -2.401428336517859, "value_error_coef": 0.5388}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.000998, "delta": 0.000998}},
#     {"target": 0.1784, "params": {'dim_hidden_a2c': 7.976134472831655,'dim_hidden_lstm': 7.535275376620803, "entropy_error_coef": 0.05111, "lstm_learning_rate": -2.0564560228465454, "value_error_coef": 0.02739}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.000998, "delta": 0.0}},
#     {"target": 0.2878, "params": {'dim_hidden_a2c': 8.509379148914688,'dim_hidden_lstm': 6.971543553950772, "entropy_error_coef": 0.1397, "lstm_learning_rate": -2.8523236757589014, "value_error_coef": 0.1981}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.001999, "delta": 0.001001}},
#     {"target": 0.184, "params": {'dim_hidden_a2c': 8.72349030214343,'dim_hidden_lstm': 7.959422420093674, "entropy_error_coef": 0.07836, "lstm_learning_rate": -2.159642940796644, "value_error_coef": 0.8764}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.001999, "delta": 0.0}},
#     {"target": 0.1967, "params": {'dim_hidden_a2c': 8.507001732764124,'dim_hidden_lstm': 6.962317654942308, "entropy_error_coef": 0.1183, "lstm_learning_rate": -2.47729500726525, "value_error_coef": 0.0559}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.002999, "delta": 0.001}},
#     {"target": 0.1785413333333334, "params": {'dim_hidden_a2c': 7.970542275711223,'dim_hidden_lstm': 7.595090999802308, "entropy_error_coef": 2.859370433622166e-05, "lstm_learning_rate": -4.0930022821044805, "value_error_coef": 0.14675589081711304}, "datetime": {"datetime": "2022-06-24 17:53:59", "elapsed": 4036.367885, "delta": 4036.364886}},
#     {"target": 0.1794506666666668, "params": {'dim_hidden_a2c': 6.719288742099458,'dim_hidden_lstm': 6.204028928407345, "entropy_error_coef": 0.08639018176076194, "lstm_learning_rate": -3.80969757730799, "value_error_coef": 0.538816734003357}, "datetime": {"datetime": "2022-06-24 18:53:09", "elapsed": 7586.800449, "delta": 3550.432564}},
#     {"target": 0.17880800000000013, "params": {'dim_hidden_a2c': 8.509258171883545,'dim_hidden_lstm': 6.971270803975987, "entropy_error_coef": 0.13967245711143791, "lstm_learning_rate": -4.578839184214298, "value_error_coef": 0.1981014890848788}, "datetime": {"datetime": "2022-06-25 00:58:52", "elapsed": 3690.115813, "delta": 3690.114795}},
#     {"target": 0.1900160000000001, "params": {'dim_hidden_a2c': 7.631937954129556,'dim_hidden_lstm': 7.073559441674648, "entropy_error_coef": 0.14653443867158242, "lstm_learning_rate": -2.770736961792505, "value_error_coef": 0.47419758533659195}, "datetime": {"datetime": "2022-06-25 02:00:11", "elapsed": 7368.79629, "delta": 3678.680477}},
#     {"target": 0.1881920000000001, "params": {'dim_hidden_a2c': 8.49979316564652,'dim_hidden_lstm': 6.9745063250036825, "entropy_error_coef": 0.10399705748487956, "lstm_learning_rate": -2.4802976731528377, "value_error_coef": 0.9003656997697922}, "datetime": {"datetime": "2022-06-25 03:01:29", "elapsed": 11046.485084, "delta": 3677.688794}},
#     {"target": 0.1849786666666668, "params": {'dim_hidden_a2c': 7.8950477155939875,'dim_hidden_lstm': 7.903089465945223, "entropy_error_coef": 0.21219179505158609, "lstm_learning_rate": -3.2403981549142262, "value_error_coef": 0.2193865132025623}, "datetime": {"datetime": "2022-06-25 04:25:12", "elapsed": 16070.044709, "delta": 5023.559625}},
# ]
# import pprint
# import math
# for dic in a:
#     for k,v in dic.items():
#         if k == 'params':
#             v['dim_hidden_a2c'] = math.log2(v['dim_hidden_a2c'])
#             v['dim_hidden_lstm'] = math.log2(v['dim_hidden_lstm'])
# pprint.pprint(a)
# a = {'a':1, 'b':2}
# c = {}
# print(c)
# for k,v in a.items():
#     c[k] = c.get(k,0) + a[k]
# print(c)
# b = {'a':2, 'b':4}
# for k,v in b.items():
#     c[k] = c[k] + b[k]
# print(c)

# from operator import index
# import pandas as pd
# a = {"a":(1,2), "b":(2,3), "c":(3,4)}
# df = pd.DataFrame.from_dict(a, orient = 'index')
# print(df)

# print(round(4.5), round(4.49), round(4.501))

# a = [(1,2), (3,4), (5,6)]
# b = [x[1] for x in a]
# print(b)
import numpy as np
import torch
a = torch.tensor([1,0,1,1,1], dtype=float)
b = [0,0,1,0,1]
c = [0,0,1,0,0]
idx = torch.multinomial(a, 2)
mask = torch.randint_like(idx, 0, 2)
# a2 = np.array(a)
print(a, idx, mask)
for idx1, mask1 in zip(idx, mask):
    a[idx1] *= mask1 
print(a, idx, mask)

0, 1 -> 1
0, 0 -> 0
1, 1 -> 0
1, 0 -> 1
