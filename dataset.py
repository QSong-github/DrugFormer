import csv
import os
import pickle

from datasets import Dataset


with open('./token_gene_dictionary.pickle', 'rb') as file:
    dictionary_genename_token_pair = pickle.load(file)


directory_path = './original_data'
csv.field_size_limit(500000)

def rebuilder(directory_path):
    files_list = os.listdir(directory_path)
    labels_cell = []
    samples = []
    dt = {}

    for filename in files_list:
        csv_file_path = directory_path + '/' + filename
        print(csv_file_path)

        headr = True
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            pattern = []
            # sequence level
            for row in csv_reader:
                row_data = row[0].split('\t')
                if headr:
                    for gene in row_data:
                        gene = gene.replace('"', '')
                        if gene in dictionary_genename_token_pair:
                            pattern.append(dictionary_genename_token_pair[gene])
                        else:
                            pattern.append(-99999)
                    headr = False
                    # print(pattern)
                else:
                    assert len(pattern)==len(row_data)
                    seq_pattern_order_id_EXPscore = []
                    # token level
                    for i in range(len(row_data)):
                        if i==0:
                            pass
                        elif i==1:
                            if 'sensitive' in row_data[i]:
                                labels_cell.append(1)
                            elif 'resistant' in row_data[i]:
                                labels_cell.append(0)
                        else:
                            if row_data[i]=='0':
                                pass
                            else:
                                if pattern[i]==-99999: # none token
                                    pass
                                else:
                                    seq_pattern_order_id_EXPscore.append((pattern[i],row_data[i]))

                    sorted_seq_pattern = sorted(seq_pattern_order_id_EXPscore, key=lambda x: x[1], reverse=True)
                    sample = [item[0] for item in sorted_seq_pattern]


                    # only keep 2048 tokens
                    while len(sample)<2048:
                        sample.append(0)  # pad
                    assert len(sample)>=2048
                    sample =sample[:2048]

                    samples.append(sample)


    dt['input_ids'] = samples
    dt['cell_label'] = labels_cell


    my_dataset = Dataset.from_dict(dt)
    my_dataset.save_to_disk('./cell_dt')


rebuilder(directory_path)




