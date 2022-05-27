a = ['original']
b = ['update', 'update_avg', 'update_ind_avg']
mem_update_boolean = any(x in max(a,b,key=len) for x in min(b,a,key=len))
print(mem_update_boolean)


trial_hidden_states = [(0, 1), (1, 3), (2, 1), (3, 2), (4, -1)]

# Find any multi pulls on same memory
unique_id = set([int(x[1]) for x in trial_hidden_states if x[1] != -1])
unique_id_dict = {key: [] for key in unique_id}
print(unique_id_dict)
for state, mem_id in trial_hidden_states:
    if mem_id != -1:
        unique_id_dict[mem_id].append(state)

print(unique_id_dict)
new_keys = []
no_matches = [x[0] for x in trial_hidden_states if x[1] == -1]
for x in no_matches:
    new_keys.append([x])
print(new_keys)


from sklearn.manifold import TSNE
import numpy as np
import matplotlib as plt
"""
tsne = TSNE(n_components=2).fit_transform(features)
# This is it — the result named tsne is the 2-dimensional projection of the 2048-dimensional features. 
# n_components=2 means that we reduce the dimensions to two. Here we use the default values of all 
# the other hyperparameters of t-SNE used in sklearn.

# Okay, we got the t-SNE — now let’s visualize the results on a plot. 
# First, we’ll normalize the points so they are in [0; 1] range.
# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)
# Now let’s plot the 2D points, each in a color corresponding to its class label.
# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)
# # for every class, we'll add a scatter plot separately
# for label in colors_per_class:
#     # find the samples of the current class in the data
#     indices = [i for i, l in enumerate(labels) if l == label]
#     # extract the coordinates of the points of this class only
#     current_tx = np.take(tx, indices)
#     current_ty = np.take(ty, indices)
#     # convert the class color to matplotlib format
#     color = np.array(colors_per_class[label], dtype=np.float) / 255
#     # add a scatter plot with the corresponding color and label

#     ax.scatter(current_tx, current_ty, c=color, label=label)

# # build a legend using the labels we set previously

# ax.legend(loc='best')
# finally, show the plot
plt.show()
"""
