import torch
from dataloader import KfoldDataset, dataloader
import argparse
from tqdm import tqdm
from model import CellDSFormer, SVM
from transformers import BertTokenizer
from torch import optim
from tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score, AP_score, AMI, ARI
import pickle
def prepare():
    parser = argparse.ArgumentParser(description='AI4Bio')
    parser.add_argument('--ep_num', type=int, default=3, help='epoch number of training')
    parser.add_argument('--train_batch_size', type=int, default=12, help='')
    parser.add_argument('--test_batch_size', type=int, default=24, help='')
    parser.add_argument('--data_path', type=str, default='./subdt', help='subdt is just for test, plz use cell_dt')  # cell_dt
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu', help='')

    parser.add_argument('--lr', type=int, default=0.0001, help='')
    parser.add_argument('--folds', type=int, default=5, help='')
    parser.add_argument('--seq_length', type=int, default=2048, help='')
    parser.add_argument('--d_model', type=int, default=256, help='')
    parser.add_argument('--num_heads', type=int, default=8, help='')
    parser.add_argument('--num_heads_gat', type=int, default=3, help='')
    parser.add_argument('--d_ff', type=int, default=1024, help='')
    parser.add_argument('--vocab_size', type=int, default=18628, help='')
    parser.add_argument('--init_node_f', type=int, default=101, help='')
    parser.add_argument('--node_f', type=int, default=256, help='')

    args = parser.parse_args()


    with open('Gdata.pickle', 'rb') as file:
        Gdata = pickle.load(file)

    trdt_list, tedt_list = KfoldDataset(args.data_path, args.folds)


    return args, trdt_list, tedt_list, Gdata,



def run():
    args, trdt_list, tedt_list, Gdata = prepare()

    Gdata = Gdata.to(args.device)
    for f in [4, 3, 1, 2, 0]:
        model = CellDSFormer(args)
        #model = SVM(args)

        loss_function = torch.nn.CrossEntropyLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        model = model.to(args.device)

        train_data_loader, test_data_loader = dataloader(current_fold=f, train_list=trdt_list, test_list=tedt_list,
                                                         tr_bs=args.train_batch_size, te_bs=args.test_batch_size)

        for epoch in range(args.ep_num):
            loss_sum = 0
            Acc = []
            F1 = []
            AUROC = []
            Precision = []
            Recall = []
            APscore = []
            Ami = []
            Ari = []

            with tqdm(train_data_loader, ncols=80, position=0, leave=True) as batches:
                for b in batches:  # sample
                    input_ids, labels = b  # batch_size*seq_len
                    if torch.all(torch.logical_or(torch.all(labels == torch.tensor([1, 0])), torch.all(labels == torch.tensor([0, 1]))))==True:
                        continue
                    input_ids = input_ids.to(args.device)
                    labels = labels.to(args.device)
                    pred = model(input_ids, Gdata)
                    labels = labels.float()
                    pred = pred.float()
                    loss = loss_function(pred, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_sum = loss_sum + loss
                    acc = Accuracy_score(pred, labels)
                    f1 = F1_score(pred, labels)
                    aur = AUROC_score(pred, labels)
                    pre = Precision_score(pred, labels)
                    rcl = Recall_score(pred, labels)
                    aps = AP_score(pred, labels)
                    ami = AMI(pred, labels)
                    ari = ARI(pred, labels)


                    Acc.append(acc)
                    F1.append(f1)
                    AUROC.append(aur)
                    Precision.append(pre)
                    Recall.append(rcl)
                    APscore.append(aps)
                    Ami.append(ami)
                    Ari.append(ari)
                print('Training epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum, 'Accuracy:', sum(Acc) / len(Acc),
                      'AUROC:', sum(AUROC) / len(AUROC),
                      'Precision:', sum(Precision) / len(Precision), 'Recall:', sum(Recall) / len(Recall), 'F1:',
                      sum(F1) / len(F1),
                      'APscore:', sum(APscore) / len(APscore), 'Ami:', sum(Ami) / len(Ami), 'Ari:', sum(Ari) / len(Ari))

            loss_sum = 0
            Acc = []
            F1 = []
            AUROC = []
            Precision = []
            Recall = []
            with torch.no_grad():
                with tqdm(test_data_loader, ncols=80, position=0, leave=True) as batches:
                    for b in batches:  # sample
                        input_ids, labels = b  # batch_size*seq_len
                        if torch.all(torch.logical_or(torch.all(labels == torch.tensor([1, 0])),
                                                      torch.all(labels == torch.tensor([0, 1])))) == True:
                            continue
                        input_ids = input_ids.to(args.device)
                        labels = labels.to(args.device)
                        pred = model(input_ids, Gdata)
                        labels = labels.float()
                        pred = pred.float()
                        loss = loss_function(pred, labels)

                        loss_sum = loss_sum + loss
                        acc = Accuracy_score(pred, labels)
                        f1 = F1_score(pred, labels)
                        aur = AUROC_score(pred, labels)
                        pre = Precision_score(pred, labels)
                        rcl = Recall_score(pred, labels)
                        aps = AP_score(pred, labels)
                        ami = AMI(pred, labels)
                        ari = ARI(pred, labels)


                        Acc.append(acc)
                        F1.append(f1)
                        AUROC.append(aur)
                        Precision.append(pre)
                        Recall.append(rcl)
                        APscore.append(aps)
                        Ami.append(ami)
                        Ari.append(ari)
                    print('Testing epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum, 'Accuracy:',
                          sum(Acc) / len(Acc), 'AUROC:', sum(AUROC) / len(AUROC),
                          'Precision:', sum(Precision) / len(Precision), 'Recall:', sum(Recall) / len(Recall), 'F1:',
                          sum(F1) / len(F1),
                          'APscore:', sum(APscore) / len(APscore), 'Ami:',sum(Ami) / len(Ami), 'Ari:',sum(Ari)/ len(Ari))

        torch.save(model.state_dict(), './model_save/model.ckpt')




if __name__ == '__main__':
    run()


