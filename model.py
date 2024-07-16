from torch import nn
from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F


class CellDSFormer(nn.Module):
    def __init__(self, args):
        super(CellDSFormer, self).__init__()
        self.args =args

        self.word_embed_layer = nn.Embedding(self.args.vocab_size, self.args.d_model)
        self.pos_embed_layer = nn.Embedding(self.args.seq_length, self.args.d_model)

        self.GAT1 = GAT(in_dims=self.args.init_node_f, out_dims=self.args.d_model, num_heads=self.args.num_heads_gat)
        self.GAT2 = GAT(in_dims=self.args.node_f, out_dims=self.args.d_model, num_heads=self.args.num_heads_gat)

        self.Transformer1 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer2 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer3 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer4 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer5 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)
        self.Transformer6 = nn.TransformerEncoderLayer(
                d_model=self.args.d_model, nhead=self.args.num_heads, dim_feedforward=self.args.d_ff, batch_first=True)

        self.aggregator1 = nn.Sequential(
            nn.Linear(self.args.d_model*2, self.args.d_model),
            nn.ReLU()
            )

        self.aggregator2 = nn.Sequential(
            nn.Linear(self.args.d_model*2, self.args.d_model),
            nn.ReLU()
            )

        self.bottleneck_layer = nn.Linear(self.args.seq_length*self.args.d_model, 32)
        self.classfier = nn.Linear(32, 2)


    def GFcandidate(self,seq, gf):
        seq = seq.long()
        seq_gf_pair = gf[seq]

        return seq_gf_pair


    def forward(self, seq, Gdata):
        seq_pos = torch.arange(2048)
        seq_pos = seq_pos.reshape(1, 2048).to(self.args.device)
        seq_word_embed = self.word_embed_layer(seq)
        seq_pos_embed = self.pos_embed_layer(seq_pos)
        embed = seq_word_embed + seq_pos_embed

        embed1 = self.Transformer1(embed)
        graphf1 = self.GAT1(Gdata['x'],Gdata['edge_index'], Gdata['edge_weights'])
        seq_gf_pair1 = self.GFcandidate(seq,graphf1)

        fused_embed1 = torch.cat((embed1, seq_gf_pair1), dim=2)
        fused_embed1 = self.aggregator1(fused_embed1)


        embed2 = self.Transformer2(fused_embed1)
        embed3 = self.Transformer3(embed2)
        embed4 = self.Transformer4(embed3)
        embed5 = self.Transformer5(embed4)
        embed6 = self.Transformer6(embed5)
        graphf2 = self.GAT2(graphf1, Gdata['edge_index'], Gdata['edge_weights'])
        seq_gf_pair2 = self.GFcandidate(seq, graphf2)

        fused_embed2 = torch.cat((embed6, seq_gf_pair2), dim=2)
        fused_embed2 = self.aggregator2(fused_embed2)



        x = fused_embed2.view(fused_embed2.size(0), -1)
        x = self.bottleneck_layer(x)
        x = self.classfier(x)

        return x




class GAT(torch.nn.Module):
    def __init__(self, in_dims, out_dims, num_heads):
        super(GAT, self).__init__()
        self.gat = GATConv(in_dims, out_dims, heads=num_heads)
        self.l = nn.Linear(out_dims*num_heads, out_dims)


    def forward(self, x, edge_index, edge_weights):
        x = self.gat(x, edge_index, edge_weights)
        x = F.relu(x)
        x = F.dropout(x, p=0.1)
        x =self.l(x)
        return x


class SVM(nn.Module):
    def __init__(self, args):
        super(SVM, self).__init__()
        self.input_size = args.seq_length


        self.fc = nn.Linear(self.input_size, 2)

    def forward(self, seq, Gdata):


        output = self.fc(seq.float())

        return output
