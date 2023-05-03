from preprocessing.csv_preprocessing import return_train_ui_matrix, return_valid_ui_data
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import numpy as np 

train_ui_data, train_ui_matrix = return_train_ui_matrix()
valid_ui_data = return_valid_ui_data()

#####################################################################################################
# Item based cosine-similarity
#####################################################################################################

LATENT_DIM = [10, 20, 50, 100, 1000]

for n_component in LATENT_DIM:
    SVD = TruncatedSVD(n_components = n_component)
    matrix = SVD.fit_transform(train_ui_matrix.T)
    item_based_collabor = np.corrcoef(matrix)

    user_item_sim = dict()
    for i in range(train_ui_data.shape[0]):
        user, item = int(train_ui_data[i][0]), int(train_ui_data[i][1])
        if not user in user_item_sim.keys():
            user_item_sim[user] = dict()
            user_item_sim[user]['true'] = []
        user_item_sim[user]['true'].append(item)

    item_cossim_total = np.sum(item_based_collabor)

    for user in tqdm(user_item_sim.keys()):
        num_candidate, num_not_candidate = len(user_item_sim[user]['true']), 27694 - len(user_item_sim[user]['true'])

        candidate_item = item_based_collabor[user_item_sim[user]['true'], :][:, user_item_sim[user]['true']]
        item_candidate_cossim_total = np.sum(candidate_item)
        item_not_candidate_cossim_total = item_cossim_total - item_candidate_cossim_total

        user_item_sim[user]['true_sim_min'] = np.min(candidate_item)
        user_item_sim[user]['true_sim_mean'] = (item_candidate_cossim_total - num_candidate) / (num_candidate * (num_candidate - 1))
        user_item_sim[user]['fake_sim_mean'] = (item_not_candidate_cossim_total - num_not_candidate) / (num_not_candidate * (num_not_candidate - 1))

    predicted = []
    for i in range(valid_ui_data.shape[0]):
        user, item, answer = int(valid_ui_data[i][0]), int(valid_ui_data[i][1]), int(valid_ui_data[i][2])
        target_item_sim = item_based_collabor[item, user_item_sim[user]['true']]
        target_sim_mean = np.mean(target_item_sim)
        target_sim_log_mean = np.log(target_sim_mean)

        is_over_min = int(target_sim_mean > user_item_sim[user]['true_sim_min'])
        is_similar_to_mean = int(abs(user_item_sim[user]['true_sim_mean'] - target_sim_mean) < abs(user_item_sim[user]['fake_sim_mean'] - target_sim_mean))
        is_similar_to_log_mean = int(abs(np.log(user_item_sim[user]['true_sim_mean']) - target_sim_log_mean) < abs(np.log(user_item_sim[user]['fake_sim_mean']) - target_sim_log_mean))

        correct = [is_over_min == answer, is_similar_to_mean == answer, is_similar_to_log_mean == answer]
        predicted.append(correct)

    item_based_accuracy = np.mean(np.array(predicted), axis=0)
    print(f"Item-based matrix-factorization with {n_component}-dim accuracy: {item_based_accuracy}")

#####################################################################################################
# User based cosine-similarity
#####################################################################################################
