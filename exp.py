from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import csv
from dataloader import KfoldDataset, dataloader
import argparse
from tqdm import tqdm
from model import CellDSFormer
from torch import optim
from sklearn.metrics import accuracy_score, f1_score

import pickle
import torch
import torch.nn.functional as F
class BioDataset(Dataset):
    def __init__(self, f_path):
        super(BioDataset, self).__init__()
        print('loading dataset...')
        self.dataset = load_from_disk(f_path)


        self.tokens = self.dataset['input_ids']
        self.labels = self.dataset['cell_label']


        self.length = len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item], self.labels[item]

    def __len__(self):
        return self.length


def bio_collate_fn(batches):
    batch_token = []
    batch_label = []
    for batch in batches:
        batch_token.append(torch.tensor(batch[0]))
        batch_label.append(torch.tensor(batch[1]))

    batch_token = torch.stack(batch_token)
    batch_label = torch.stack(batch_label)



    return batch_token,batch_label







def prepare():
    parser = argparse.ArgumentParser(description='AI4Bio')

    # training parameters
    parser.add_argument('--ep_num', type=int, default=3, help='epoch number of training')
    parser.add_argument('--train_batch_size', type=int, default=32, help='')
    parser.add_argument('--test_batch_size', type=int, default=64, help='')
    parser.add_argument('--data_path', type=str, default='./GSE207422_Tor_post_dt', help='')   # newdt
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='')
    #parser.add_argument('--device', type=str, default='cpu', help='')

    parser.add_argument('--seq_length', type=int, default=2048, help='')
    parser.add_argument('--d_model', type=int, default=256, help='')
    parser.add_argument('--num_heads', type=int, default=8, help='')
    parser.add_argument('--num_heads_gat', type=int, default=3, help='')
    parser.add_argument('--d_ff', type=int, default=1024, help='')
    parser.add_argument('--vocab_size', type=int, default=18628, help='')
    parser.add_argument('--init_node_f', type=int, default=101, help='')
    parser.add_argument('--node_f', type=int, default=256, help='')

    args = parser.parse_args()


    return args

def F1_score(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = labels.cpu()
    F1 = f1_score(max_prob_index_pred, max_prob_index_labels)

    return F1

def run():
    args= prepare()

    biodataset = BioDataset('./GSE162117_mal_pre_countsMatrix_dt')  #GSE162117_mal_pre_countsMatrix_dt   GSE161801_IMiD_mal_pre_countsMatrix_dt
    test_data_loader = DataLoader(dataset=biodataset, batch_size=64, shuffle=True,
                                  collate_fn=bio_collate_fn)

    model = CellDSFormer(args)

    model.load_state_dict(torch.load('./model_save/model.ckpt'))
    model.eval()
    model = model.to(args.device)

    F1 = []
    out = []
    with open('Gdata.pickle', 'rb') as file:
        Gdata = pickle.load(file)
    Gdata = Gdata.to(args.device)
    with (torch.no_grad()):
        with tqdm(test_data_loader, ncols=80, position=0, leave=True) as batches:
            for b in batches:  # sample
                input_ids, labels = b  # batch_size*seq_len
                input_ids = input_ids.to(args.device)
                labels = labels.to(args.device)
                preds = model(input_ids, Gdata)
                f1 = F1_score(preds.float(), labels.float())
                F1.append(f1)
                preds = torch.sigmoid(preds)

                preds = preds.tolist()

                for sample_results in preds:
                    out.append(sample_results)

    print('F1:',sum(F1) / len(F1))
    # csv_file_path = './GSE161801_IMiD_mal_pre_countsMatrix_dt.csv'
    #
    # with open(csv_file_path, mode='w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerows(out)



run()
