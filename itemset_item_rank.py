from preprocessing.csv_preprocessing import return_train_ii_matrix, return_valid_ii_data, return_ui_matrix
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import numpy as np 
from time import time

train_ui_data, train_ui_matrix = return_ui_matrix()
valid_ui_query, valid_ui_answer = return_valid_ii_data()

item_frequency = train_ui_matrix.sum(axis=0)
item_frequency = (item_frequency - np.min(item_frequency)) / (np.max(item_frequency) - np.min(item_frequency)) # normalize to [0, 1]

start_time = time()
item_based_collabor = cosine_similarity(train_ui_matrix.T)
end_time = time() - start_time
print(f'time: {end_time:.3f}s')

itemset_item_dict = dict()
for i in range(valid_ui_query.shape[0]):
    itemset_id, item_id = valid_ui_query[i][0], valid_ui_query[i][1]
    if not valid_ui_query[i][0] in itemset_item_dict.keys():
        itemset_item_dict[itemset_id] = []
    itemset_item_dict[itemset_id].append(item_id)

freq_scale = [i * 0.1 for i in range(11)]
for s in freq_scale:
    acc, rank = 0, 0
    for i in range(valid_ui_answer.shape[0]):
        incom_items = list(map(int, itemset_item_dict[valid_ui_answer[i][0]]))
        incom_items_collabor = item_based_collabor[incom_items, :]
        incom_items_collabor[:, incom_items] = 0
        mean_sim_itemset = np.mean(incom_items_collabor, axis=0)
        mean_sim_itemset = (mean_sim_itemset - np.min(mean_sim_itemset)) / (np.max(mean_sim_itemset) - np.min(mean_sim_itemset)) # normalize to [0, 1]
        mean_sim_itemset = mean_sim_itemset * (1 - s) + item_frequency * s
        sorted_item_idx = mean_sim_itemset.argsort()[::-1]
        
        if valid_ui_answer[i][1] in sorted_item_idx[:100]:
            acc += 1
            rank += np.where(sorted_item_idx[:100] == valid_ui_answer[i][1])[0][0]
        else:
            rank += 101

    acc /= valid_ui_answer.shape[0]; rank /= valid_ui_answer.shape[0]
    print(f"similarity scale: {s}, Accuracy: {acc:.5f}, Average of Rank: {rank:.3f}")