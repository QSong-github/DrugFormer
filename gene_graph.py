import csv
import os
import pickle
import torch
from datasets import Dataset
from datasets import load_from_disk
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

def dictionary():
    d = ['1-Mar', 'MARCH8', 'MARCH5', '6-Mar', '3-Mar', 'MARCH1', '5-Mar', 'MARCH6', '2-Mar', '10-Mar', 'MARCH11', '11-Mar', '7-Mar',
         'MARCH9', 'MARCH4', '9-Mar', 'MARCH10', '15-Sep', 'MARCH7', '8-Mar', '4-Mar', 'MARCH3', '1-Dec', 'SEP15', 'MARCH2', 'DEC1']
    with open('./Collins_rCNV_2022.dosage_sensitivity_scores/rCNV.gene_scores.tsv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = True
        i = 0
        token_gene_dictionary = {}
        for row in csv_reader:
            if header == True:
                header = False
                continue
            row_data = row[0].split('\t')
            gene = row_data[0]
            if gene not in d:
                token_gene_dictionary[gene] = i
                i = i + 1


    file_path = 'token_gene_dictionary.pickle'
    with open(file_path, 'wb') as file:
        pickle.dump(token_gene_dictionary, file)

    return token_gene_dictionary



def dictionary_meanScores_token_pair_building(dictionary):
    dictionary_meanScores_token_pair = {}
    with open('./Collins_rCNV_2022.dosage_sensitivity_scores/rCNV.gene_scores.tsv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = True
        for row in csv_reader:
            if header == True:
                header = False
                continue
            row_data = row[0].split('\t')
            gene = row_data[0]
            if gene in dictionary:
                score1 = float(row_data[1])
                score2= float(row_data[2])
                mean = (score1 + score2) / 2
                dictionary_meanScores_token_pair[dictionary[gene]] = mean

    l_score = []
    for i in range(len(dictionary_meanScores_token_pair)):
        l_score.append(dictionary_meanScores_token_pair[i])

    edge_index = []
    edge_weight = []
    for i in range(len(l_score)):
        for j in range(i,len(l_score)):
            edge_value = 1 - abs(l_score[i] - l_score[j])
            if edge_value>0.999:
                edge_index.append([i,j])
                edge_weight.append(edge_value)


    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_weight = torch.tensor(edge_weight)
    return edge_index, edge_weight



def dictionary_eigenfeatures_token_pair_building(dictionary):
    dictionary_eigenfeatures_token_pair = {}
    with open('./Collins_rCNV_2022.gene_features_matrix/Collins_rCNV_2022.gene_eigenfeatures1.csv', 'r') as file:
        csv_reader = csv.reader(file)
        header = True
        for row in csv_reader:
            if header == True:
                header = False
                continue

            gene = row[3]
            if gene in dictionary:
                f = []
                for i in range(4,len(row)):
                    f.append(float(row[i]))
                dictionary_eigenfeatures_token_pair[dictionary[gene]] = f


    l = []
    for i in range(len(dictionary_eigenfeatures_token_pair)):
        l.append(dictionary_eigenfeatures_token_pair[i])


    node_features = torch.tensor(l)
    return node_features



token_gene_dictionary = dictionary()
edge_index, edge_weights = dictionary_meanScores_token_pair_building(token_gene_dictionary)
print(edge_index.shape)
print(edge_weights.shape)

node_features = dictionary_eigenfeatures_token_pair_building(token_gene_dictionary)
print(node_features.shape)


Gdata = Data(x=node_features, edge_index=edge_index, edge_weights=edge_weights)

file_path = 'Gdata.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(Gdata, file)
