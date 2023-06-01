from preprocessing.csv_preprocessing import return_train_ui_matrix, return_valid_ui_data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np 
from time import time
import os
import pickle as pkl

train_ui_data, train_ui_matrix = return_train_ui_matrix()
valid_ui_data = return_valid_ui_data()

if os.path.exists('./itemset_based_collabor.pkl'):
    itemset_based_collabor = pkl.load(open('./itemset_based_collabor.pkl', 'rb'))
else:
    itemset_based_collabor = cosine_similarity(train_ui_matrix.T) # from User-itemset matrix -> itemset similarity
itemset_cossim_total = np.sum(itemset_based_collabor)

item_frequency = train_ui_matrix.sum(axis=0)
top_itemset = item_frequency.argsort()[::-1][:500]

item_frequency = (item_frequency - np.min(item_frequency)) / (np.max(item_frequency) - np.min(item_frequency)) # normalize to (0,1)
top_itemset_freq_dict = dict()
for i in top_itemset:
    top_itemset_freq_dict[i] = item_frequency[i]

start_time = time()
user_item_sim = dict()
for i in range(train_ui_data.shape[0]):
    user, item = int(train_ui_data[i][0]), int(train_ui_data[i][1])
    if not user in user_item_sim.keys():
        user_item_sim[user] = dict()
        user_item_sim[user]['true'] = []
    user_item_sim[user]['true'].append(item)

for user in tqdm(user_item_sim.keys()):
    candidate_item = itemset_based_collabor[user_item_sim[user]['true'], :][:, user_item_sim[user]['true']]
    user_item_sim[user]['true_sim_min'] = np.min(candidate_item)

scale = [i*0.01 for i in range(51)]
preds, answers = [[] for _ in range(len(scale))], [[] for _ in range(len(scale))]

for i in tqdm(range(valid_ui_data.shape[0])):
    user, item, answer = int(valid_ui_data[i][0]), int(valid_ui_data[i][1]), int(valid_ui_data[i][2])
    target_item_sim = itemset_based_collabor[item, user_item_sim[user]['true']]
    target_sim_mean = np.mean(target_item_sim)

    for j in range(len(scale)):
        s = scale[j]
        try:
            freq = top_itemset_freq_dict[item]
        except:
            freq = 0
        score = target_sim_mean + s * freq
        is_over_min = int(score > user_item_sim[user]['true_sim_min'])

        answers[j].append(answer)
        preds[j].append(is_over_min)

for j in range(len(scale)):
    cf = confusion_matrix(answers[j], preds[j])
    acc = (cf[0][0]+cf[1][1])/len(answers[j])
    prec = precision_score(answers[j], preds[j])
    recall = recall_score(answers[j], preds[j])
    f1 = f1_score(answers[j], preds[j])
    print(f'scale:{scale[j]:.2f} -> acc:{acc:.5f}, prec:{prec:.5f}, recall:{recall:.5f}, f1:{f1:.5f}')