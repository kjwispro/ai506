from preprocessing.csv_preprocessing import return_train_ii_matrix, return_train_ui_matrix, return_valid_ii_data, return_ui_matrix
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import numpy as np 
import pickle as pkl
import os
import itertools
from time import time

start_time = time()
ui_data, ui_matrix = return_ui_matrix()
train_ui_data, train_ui_matrix = return_train_ui_matrix()
train_ii_data, train_ii_matrix = return_train_ii_matrix()
valid_ui_query, valid_ui_answer = return_valid_ii_data()

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
for i in range(valid_ui_query.shape[0]):
    itemset_id, item_id = valid_ui_query[i][0], valid_ui_query[i][1]
    if not valid_ui_query[i][0] in itemset_item_dict.keys():
        itemset_item_dict[itemset_id] = []
    itemset_item_dict[itemset_id].append(item_id)
    
end_time = time() - start_time
print(f'setting time: {end_time:.3f}s')

betas  = [i * 0.025 for i in range(5)]
gammas = [i * 0.1 for i in range(8)]
all_consts = list(itertools.product(betas, gammas))

acc, rank = [0 for _ in range(len(all_consts))], [0 for _ in range(len(all_consts))]
for i in tqdm(range(valid_ui_answer.shape[0])):
    itemset_id = valid_ui_answer[i][0]
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
    
    j = 0
    for beta, gamma in all_consts:
        alpha = 1 - beta - gamma
        score = alpha * mean_sim_item + beta * item_frequency + gamma * item_sim_by_itemset_sim 
        score[incom_items] = 0  # except for existing items
        sorted_item_idx = score.argsort()[::-1]
        
        if valid_ui_answer[i][1] in sorted_item_idx[:100]:
            acc[j] += 1
            rank[j] += np.where(sorted_item_idx[:100] == valid_ui_answer[i][1])[0][0]
        else:
            rank[j] += 101
        j += 1

total_answers = valid_ui_answer.shape[0]

for j in range(len(all_consts)):
    beta, gamma = all_consts[j]
    alpha = 1 - beta - gamma
    print(f"alpha:{alpha:.3f}, beta:{beta:.3f}, gamma:{gamma:.1f} // Accuracy: {acc[j]/total_answers:.5f}, Average of Rank: {rank[j]/total_answers:.3f}")