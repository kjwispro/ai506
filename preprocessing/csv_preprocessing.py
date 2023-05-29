import numpy as np 
from tqdm import tqdm



def return_train_ui_matrix():
    user_itemset_data = np.loadtxt('/home/ai506/dataset/user_itemset_training.csv', delimiter=',')
    matrix = np.zeros((53897, 27694))
    for i in range(user_itemset_data.shape[0]):
        pair = user_itemset_data[i]
        matrix[int(pair[0])][int(pair[1])] = 1
    
    return user_itemset_data, matrix


def return_valid_ui_data():
    query = np.loadtxt('/home/ai506/dataset/user_itemset_valid_query.csv', delimiter=',')
    answer = np.loadtxt('/home/ai506/dataset/user_itemset_valid_answer.csv', delimiter=',')
    data = np.concatenate([query, np.expand_dims(answer, axis=1)], axis=1)
    return data


def return_train_ii_matrix():
    itemset_item_data = np.loadtxt('/home/ai506/dataset/itemset_item_training.csv', delimiter=',')
    matrix = np.zeros((27694, 42653))
    for i in range(itemset_item_data.shape[0]):
        pair = itemset_item_data[i]
        matrix[int(pair[0])][int(pair[1])] = 1
    
    return itemset_item_data, matrix


def return_valid_ii_data():
    query = np.loadtxt('/home/ai506/dataset/itemset_item_valid_query.csv', delimiter=',')
    answer = np.loadtxt('/home/ai506/dataset/itemset_item_valid_answer.csv', delimiter=',')
    return query, answer

def return_ui_matrix():
    user_item_data = np.loadtxt('/home/ai506/dataset/user_item.csv', delimiter=',')
    matrix = np.zeros((53897, 42653))
    for i in range(user_item_data.shape[0]):
        pair = user_item_data[i]
        matrix[int(pair[0])][int(pair[1])] = 1
    
    return user_item_data, matrix