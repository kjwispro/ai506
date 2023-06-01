from preprocessing.csv_preprocessing import return_ui_matrix, return_train_ui_matrix, return_train_ii_matrix, return_task1_test, return_task2_test
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import os
import csv
import pickle as pkl

ui_data, ui_matrix = return_ui_matrix()
train_ui_data, train_ui_matrix = return_train_ui_matrix()
train_ii_data, train_ii_matrix = return_train_ii_matrix()
test_ui_query_task1 = return_task1_test()
test_ii_query_task2 = return_task2_test()

##### Task 1 answer #####

if os.path.exists('./itemset_based_collabor.pkl'):
    itemset_based_collabor = pkl.load(open('./itemset_based_collabor.pkl', 'rb'))
else:
    itemset_based_collabor = cosine_similarity(train_ui_matrix.T) # from User-itemset matrix -> itemset similarity
user_item_sim = dict()
for i in range(train_ui_data.shape[0]):
    user, item = int(train_ui_data[i][0]), int(train_ui_data[i][1])
    if not user in user_item_sim.keys():
        user_item_sim[user] = dict()
        user_item_sim[user]['true'] = []
    user_item_sim[user]['true'].append(item)

for user in user_item_sim.keys():
    candidate_item = itemset_based_collabor[user_item_sim[user]['true'], :][:, user_item_sim[user]['true']]
    user_item_sim[user]['true_sim_min'] = np.min(candidate_item)
    
with open('user_itemset_test_prediction.csv', 'w') as f:
    writer = csv.writer(f)
    for i in range(test_ui_query_task1.shape[0]):
        user, item = int(test_ui_query_task1[i][0]), int(test_ui_query_task1[i][1])
        target_item_sim = itemset_based_collabor[item, user_item_sim[user]['true']]
        target_sim_mean = np.mean(target_item_sim)
        predict_answer = int(target_sim_mean > user_item_sim[user]['true_sim_min'])
        writer.writerow(str(predict_answer))

##### Task 2 answer #####

item_frequency = ui_matrix.sum(axis=0)
item_frequency = (item_frequency - np.min(item_frequency)) / (np.max(item_frequency) - np.min(item_frequency)) # normalize to [0, 1]

if os.path.exists('./itemset_based_collabor.pkl'):
    itemset_based_collabor = pkl.load(open('./itemset_based_collabor.pkl', 'rb'))
else:
    itemset_based_collabor = cosine_similarity(train_ui_matrix.T) # from User-itemset matrix -> itemset similarity
if os.path.exists('./item_based_collabor.pkl'):
    item_based_collabor = pkl.load(open('./item_based_collabor.pkl', 'rb'))
else:
    item_based_collabor = cosine_similarity(ui_matrix.T) # from User-item matrix -> item similarity

train_itemset_item_dict = dict()
for i in range(train_ii_data.shape[0]):
    itemset_id, item_id = train_ii_data[i][0], train_ii_data[i][1]
    if not train_ii_data[i][0] in train_itemset_item_dict.keys():
        train_itemset_item_dict[itemset_id] = []
    train_itemset_item_dict[itemset_id].append(item_id)

itemset_item_dict = dict()
for i in range(test_ii_query_task2.shape[0]):
    itemset_id, item_id = test_ii_query_task2[i][0], test_ii_query_task2[i][1]
    if not test_ii_query_task2[i][0] in itemset_item_dict.keys():
        itemset_item_dict[itemset_id] = []
    itemset_item_dict[itemset_id].append(item_id)
    
# best accuracy factors
alpha = 0.6
beta  = 0
gamma = 0.4

with open('itemset_item_test_prediction.csv', 'w') as f:
    writer = csv.writer(f)
    for itemset_id in itemset_item_dict.keys():
        incom_items = list(map(int, itemset_item_dict[itemset_id]))
        incom_items_collabor = item_based_collabor[incom_items, :]
        mean_sim_item = np.mean(incom_items_collabor, axis=0)
        mean_sim_item = (mean_sim_item - np.min(mean_sim_item)) / (np.max(mean_sim_item) - np.min(mean_sim_item)) # normalize to [0, 1]
        
        item_sim_by_itemset_sim = np.zeros((42653,))
        itemset_similarities = itemset_based_collabor[int(itemset_id)]
        for its_key in train_itemset_item_dict.keys():
            for item_key in train_itemset_item_dict[its_key]:
                item_sim_by_itemset_sim[int(item_key)] += itemset_similarities[int(its_key)]
        itemset_similarities = (itemset_similarities - np.min(itemset_similarities)) / (np.max(itemset_similarities) - np.min(itemset_similarities)) # normalize to [0, 1]
        
        alpha = 1 - beta - gamma
        score = alpha * mean_sim_item + beta * item_frequency + gamma * item_sim_by_itemset_sim 
        score[incom_items] = 0  # except for existing items
        sorted_item_idx = list(score.argsort()[::-1])[:100]
        answer_line = [int(itemset_id)] + sorted_item_idx
        
        writer.writerow(answer_line)
        