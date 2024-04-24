import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from collections import Counter
from sklearn.metrics import roc_curve, auc, average_precision_score
import joblib


# Directory path
directory_path = "/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/games_clean.json"

# List all files in the given directory
files = [f for f in os.listdir(directory_path) if f.endswith('.json') and f.startswith('part')]

# Create an empty DataFrame
df = pd.DataFrame()

# Read each file and append it to the DataFrame
for file in files:
    file_path = os.path.join(directory_path, file)
    file_df = pd.read_json(file_path, lines=True)
    df = pd.concat([df, file_df], ignore_index=True)

# Data processing
df['UserID'] = df['steamid'].astype(int)
df['Game'] = df['name'].astype(str)
df['Hours_Played'] = df['playtime_forever'].astype('float32')

# Sorting
df.UserID = df.UserID.astype('int')
df = df.sort_values(['UserID', 'Game', 'Hours_Played'])

# Create a new DataFrame clean_df
clean_df = df.drop_duplicates(['UserID', 'Game'], keep='last').drop(['steamid', 'name', 'playtime_forever', 'appid'], axis=1)
clean_df.count()

n_users = len(clean_df.UserID.unique())
n_games = len(clean_df.Game.unique())
print('There are {0} users and {1} games in the user-game dataset.'.format(n_users, n_games))

# Calculate the sparsity of the matrix
sparsity = clean_df.shape[0] / float(n_users * n_games)
print('The sparsity of the user-game matrix is: {:.2%}'.format(sparsity))

# Establish serialized IDs for easy use

# Dictionary from user id to serialized user id
user2idx = {user: i for i, user in enumerate(clean_df.UserID.unique())}
# Dictionary from serialized user id to user id
idx2user = {i: user for user, i in user2idx.items()}

# Dictionary from game name to serialized game id
game2idx = {game: i for i, game in enumerate(clean_df.Game.unique())}
# Dictionary from serialized game id to game name
idx2game = {i: game for game, i in game2idx.items()}

# Save dictionaries for later use in PyQt5
#joblib.dump(idx2game, '/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/idx2game.pkl')
#joblib.dump(game2idx, '/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/game2idx.pkl')

# User serialized id - Game serialized id - Game duration
user_idx = clean_df['UserID'].apply(lambda x: user2idx[x]).values
game_idx = clean_df['gamesIdx'] = clean_df['Game'].apply(lambda x: game2idx[x]).values
hours = clean_df['Hours_Played'].values
# Save game duration matrix
hours_save = np.zeros(shape=(n_users, n_games))
for i in range(len(user_idx)):
    hours_save[user_idx[i], game_idx[i]] = hours[i]
#joblib.dump(hours_save, '/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/hours.pkl')

# Establish sparse matrix to store large dataset
# Confidence matrix:
# Increase confidence based on game duration, with a minimum of 1

zero_matrix = np.zeros(shape=(n_users, n_games))
# Purchase matrix
user_game_pref = zero_matrix.copy()
user_game_pref[user_idx, game_idx] = 1
# Save purchase matrix
#joblib.dump(user_game_pref, '/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/buy.pkl')
# Confidence matrix
user_game_interactions = zero_matrix.copy()
user_game_interactions[user_idx, game_idx] = hours + 1

k = 5

# For each user, calculate the number of games they purchased
purchase_counts = np.apply_along_axis(np.bincount, 1, user_game_pref.astype(int))
buyers_idx = np.where(purchase_counts[:, 1] >= 2 * k)[0]  # Set of buyers who purchased more than 2*k games
print('{0} players have purchased at least {1} games'.format(len(buyers_idx), 2 * k))
# Save list of effective buyers
#joblib.dump(buyers_idx, '/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/buyers.pkl')

test_frac = 0.2  # 10% of data for validation, 10% of data for testing
test_users_idx = np.random.choice(buyers_idx,
                                  size=int(np.ceil(len(buyers_idx) * test_frac)),
                                  replace=False)
val_users_idx = test_users_idx[:int(len(test_users_idx) / 2)]
test_users_idx = test_users_idx[int(len(test_users_idx) / 2):]

# Mask k games in the training set for each user
def data_process(dat, train, test, user_idx, k):
    for user in user_idx:
        purchases = np.where(dat[user, :] == 1)[0]
        mask = np.random.choice(purchases, size=k, replace=False)
        train[user, mask] = 0
        test[user, mask] = dat[user, mask]
    return train, test

train_matrix = user_game_pref.copy()
test_matrix = zero_matrix.copy()
val_matrix = zero_matrix.copy()

train_matrix, val_matrix = data_process(user_game_pref, train_matrix, val_matrix, val_users_idx, k)
train_matrix, test_matrix = data_process(user_game_pref, train_matrix, test_matrix, test_users_idx, k)

test_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]

train_matrix[test_users_idx[0], test_matrix[test_users_idx[0], :].nonzero()[0]]

tf.reset_default_graph()

# Preference matrix
pref = tf.placeholder(tf.float32, (n_users, n_games))
# Game time matrix
interactions = tf.placeholder(tf.float32, (n_users, n_games))
user_idx = tf.placeholder(tf.int32, (None))

n_features = 30  # Set the number of hidden features to 30

# X matrix (User - Hidden features) represents user latent preferences
X = tf.Variable(tf.truncated_normal([n_users, n_features], mean=0, stddev=0.05), dtype=tf.float32, name='X')
# Y matrix (Game - Hidden features) represents game latent features
Y = tf.Variable(tf.truncated_normal([n_games, n_features], mean=0, stddev=0.05), dtype=tf.float32, name='Y')

# Confidence parameter initialization
conf_alpha = tf.Variable(tf.random_uniform([1], 0, 1))

# Initialize user bias
user_bias = tf.Variable(tf.truncated_normal([n_users, 1], stddev=0.2))

# Concatenate vectors to the user matrix
X_plus_bias = tf.concat([X,
                         user_bias,
                         tf.ones((n_users, 1), dtype=tf.float32)],
                        axis=1)

# Initialize item bias
item_bias = tf.Variable(tf.truncated_normal([n_games, 1], stddev=0.2))

# Concatenate vectors to the game matrix
Y_plus_bias = tf.concat([Y,
                         tf.ones((n_games, 1), dtype=tf.float32),
                         item_bias],
                        axis=1)

# Determine result score matrix through matrix multiplication
pred_pref = tf.matmul(X_plus_bias, Y_plus_bias, transpose_b=True)

# Construct confidence matrix using game time and alpha parameters
conf = 1 + conf_alpha * interactions

# Loss function
cost = tf.reduce_sum(tf.multiply(conf, tf.square(tf.subtract(pref, pred_pref))))
l2_sqr = tf.nn.l2_loss(X) + tf.nn.l2_loss(Y) + tf.nn.l2_loss(user_bias) + tf.nn.l2_loss(item_bias)
lambda_c = 0.01
loss = cost + lambda_c * l2_sqr

# Optimizer with gradient descent algorithm
lr = 0.05
optimize = tf.train.AdagradOptimizer(learning_rate=lr).minimize(loss)


# Accuracy calculation optimization, merge base game and DLC as the same game
def precision_dlc(recommendations, labels):
    # Split recommended games by word
    recommendations_split = []
    # Split purchased games by word
    labels_split = []
    for label in labels:
        labels_split.append(idx2game[label].split())
    for game in recommendations:
        recommendations_split.append(idx2game[game].split())

    count = 0
    for game in recommendations_split:
        for label in labels_split:
            # When the recommended game overlaps with the purchased game by more than a threshold, consider them the same game
            if (len(set(game) & set(label)) / min(len(game), len(label))) > 0.2:
                count += 1
                break

    return float(count / len(recommendations))

# Select top k from the predicted list
def top_k_precision(pred, mat, k, user_idx):
    precisions = []
    for user in user_idx:
        rec = np.argsort(-pred[user, :])
        # Select top k games with the highest recommendation scores
        top_k = rec[:k]
        labels = mat[user, :].nonzero()[0]
        # Calculate accuracy of recommendation compared to actual purchased games and return
        precision = precision_dlc(top_k, labels)
        precisions.append(precision)
    return np.mean(precisions)


iterations = 500
# Data for plotting: loss, training set accuracy
fig_loss = np.zeros([iterations])
fig_train_precision = np.zeros([iterations])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        sess.run(optimize, feed_dict={pref: train_matrix,
                                      interactions: user_game_interactions})
        if i % 10 == 0:
            mod_loss = sess.run(loss, feed_dict={pref: train_matrix,
                                                 interactions: user_game_interactions})
            mod_pred = pred_pref.eval()
            train_precision = top_k_precision(mod_pred, train_matrix, k, val_users_idx)
            val_precision = top_k_precision(mod_pred, val_matrix, k, val_users_idx)
            print('Current progress: {}...'.format(i),
                  'Loss: {:.2f}...'.format(mod_loss),
                  'Training accuracy: {:.3f}...'.format(train_precision),
                  'Validation accuracy: {:.3f}'.format(val_precision))
        fig_loss[i] = sess.run(loss, feed_dict={pref: train_matrix,
                                                interactions: user_game_interactions})
        fig_train_precision[i] = top_k_precision(mod_pred, train_matrix, k, val_users_idx)
    rec = pred_pref.eval()
    test_precision = top_k_precision(rec, test_matrix, k, test_users_idx)
    print('\n')
    print('Model completed, accuracy: {:.3f}'.format(test_precision))

n_examples = 5
users = np.random.choice(test_users_idx, size=n_examples, replace=False)
rec_games = np.argsort(-rec)

for user in users:
    purchase_history = np.where(train_matrix[user, :] != 0)[0]
    recommendations = rec_games[user, :]
    new_recommendations = recommendations[~np.in1d(recommendations, purchase_history)][:k]

    print('Recommended games for user ID {0}:'.format(idx2user[user]))
    print(', '.join([idx2game[game] for game in new_recommendations]))
    print('User\'s actual purchased games:')
    print(', '.join([idx2game[game] for game in np.where(test_matrix[user, :] != 0)[0]]))
    print('Precision: {:.2f}%'.format(100 * precision_dlc(new_recommendations, np.where(test_matrix[user, :] != 0)[0])))
    print('\n')

# Save the trained rating matrix
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     joblib.dump(pred_pref.eval(), '/Users/a1234/Desktop/workspace/Steam_Recommendation_System_777_tp/Processed_Data/rec.pkl')

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial']

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.plot(np.arange(iterations), fig_loss, label='Loss')
lns2 = ax2.plot(np.arange(iterations), fig_train_precision, 'r', label='Train Accuracy')
ax1.set_xlabel('Training Iterations')
ax1.set_ylabel('Training Loss')
ax2.set_ylabel('Training Accuracy')
# Merge legends
lns = lns1 + lns2
labels = ['Loss', 'Accuracy']
plt.legend(lns, labels, loc=7)
plt.show()
