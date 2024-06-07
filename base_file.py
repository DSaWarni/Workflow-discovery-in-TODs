import torch
import json
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import transformers
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch_geometric as tg
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.data import Data
import torch_geometric
import torch
import networkx as nx
import numpy as np
from copy import deepcopy
import random
seed = 17
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model_base(nn.Module):
    def __init__(self):
        super(Model_base, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(1536,31)
        self.graph_embeds = GCNConv(in_channels = -1, out_channels= 768)

    def forward(self, input_ids = None, attention_mask = None, graph_embeddings = None, graph = None, use_graph_only = False):
        if use_graph_only:
            graph = self.graph_embeds(graph.x, graph.edge_index)
            return graph
        else:
            _,o2 = self.bert_model(input_ids, attention_mask = attention_mask, return_dict = False)
            o2 = torch.cat([o2.squeeze(), graph_embeddings.mean(dim = 0)])
            out = self.out(o2)
            return out


def bert_action_embedding(action_label_map, semantic_mapping, tokenizer, model):
    action_embeddings = torch.zeros([len(action_label_map), 768]).to(device)
    tokenizer = tokenizer
    model = model
    for action in action_label_map:
        semantic = semantic_mapping[action]
        encoded_dict = tokenizer(
            semantic,
            add_special_tokens = True,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt',
        ).to(device)
        output = model(**encoded_dict)
        action_embeddings[action_label_map[action]] = output[1]
    return action_embeddings



def graph_save(discovered_graph, intent, epoch, type_ = None):
    G = torch_geometric.utils.to_networkx(discovered_graph, remove_self_loops = True)
    G.remove_nodes_from(list(nx.isolates(G)))
    if type_ == 1:
        name = 'discovered_graph'
    return None


def graph_updater(new_node, graph, last_node = None, flag = True):
    if flag is True:
        new_edges = torch.cat([graph.edge_index, torch.tensor([[0],[0]]).to(device)], dim = 1)
        graph = Data(x = graph.x, edge_index = new_edges).to(device)
    if new_node ==0:
        return graph
    if new_node != 0:
        new_edges = torch.cat([graph.edge_index, torch.tensor([[new_node],[graph.edge_index[0][-1]]]).to(device)], dim = 1)
        graph = Data(x= graph.x, edge_index = new_edges).to(device)
        return graph
    else:
        return graph

def context_cluster(conversation, action_label_map, gamma, tokenizer):
    actions = conversation['action']
    label = [action_label_map[i] for i in actions]
    Dialogues = conversation['Dialogue']
    dummy = ['NA']
    cluster = []
    for i in range(1,len(Dialogues)+1):
        if i < gamma:
            context = dummy*(gamma-i)+Dialogues[:i]
        else:
            context = Dialogues[i-gamma:i]
        context = ' [SEP] '.join(context)
        cluster.append(context)
    encoded_data_train = tokenizer.batch_encode_plus(
        cluster,
        add_special_tokens = True,
        retrun_attention_mask = True,
        pad_to_max_length = True,
        max_length = 256,
        return_tensors = 'pt',
    )
    input_ids_train = encoded_data_train['input_ids']
    attention_mask_train = encoded_data_train['attention_mask']
    labels = torch.tensor(label)
    data = TensorDataset(input_ids_train, attention_mask_train, labels)
    dataloader = DataLoader(data, batch_size=1)
    return dataloader



def cascading_evaluation(y_true, y_pred):
    """
    Implements the cascading evaluation metric.

    Args:
       y_true (torch.Tensor): Ground truth labels (batch_size, num_actions)
       y_pred (torch.Tensor): Predicted actions (batch_size, num_actions)

    Returns:
        float: Cascading evaluation score
    """
    num_samples = y_true.size(0)
    cascading_scores = torch.zeros(num_samples)

    for i in range(num_samples):
        gold_seq = y_true[i].nonzero(as_tuple=True)[0]  # Indices of true actions
        pred_seq = y_pred[i].nonzero(as_tuple=True)[0]  # Indices of predicted actions

        # Initialize a tensor to store correct predictions at each position
        correct = torch.zeros(len(gold_seq))

        # Loop through each position and check for correctness
        for j in range(len(gold_seq)):
            if gold_seq[:j+1].equal(pred_seq[:j+1]):
                correct[j] = 1
        cascading_scores[i] = torch.mean(correct)

    return torch.mean(cascading_scores)

import torch
def exact_match_metric(y_true, y_pred):
    """
    Implements the exact match metric.

    Args:
       y_true (torch.Tensor): Ground truth labels (batch_size, num_actions)
       y_pred (torch.Tensor): Predicted actions (batch_size, num_actions)

    Returns:
        float: Exact match score
    """

    num_samples = y_true.size(0)
    exact_match = 0

    for i in range(num_samples):
        if torch.all(y_true[i] == y_pred[i]):
            exact_match += 1

    return exact_match / num_samples



def train_epoch(model, train_data, action_label_map, loss_CE, loss_MSE, optimizer_1, device, tokenizer, semantic_mapping, bert_model, epoch):
    losses = []

    count  = 0
    correct_predictions = 0
    f1s = []
    ce = []
    em = []


    for intent_name in train_data:
        intent = train_data[intent_name]
        zero = torch.tensor([[0,0]])
        disc_edge_list = [zero.t().contiguous()]
        intent_edge_list = [zero.t().contiguous()]
        for convo in tqdm(intent['conversations']):
            action_embedding = bert_action_embedding(action_label_map, semantic_mapping, tokenizer, model=bert_model)
            d_base_edge_index = disc_edge_list[-1]
            i_base_edge_index = intent_edge_list[-1]
            intent_graph = Data(x = action_embedding.clone(), edge_index= i_base_edge_index).to(device)
            discovered_graph = Data(x=action_embedding, edge_index = d_base_edge_index).to(device)
            discovered_graph_embeddings = model(graph = discovered_graph, use_graph_only = True)
            linear_flow = None
            conversations = intent['conversations'][convo]
            clusters = context_cluster(conversations, action_label_map, 3, tokenizer)
            output = []
            pred = []
            target = []
            prev_node = 0
            real_prev_node = 0
            for index, d in enumerate(clusters):
                if index != 0:
                    flag = False
                else:
                    flag = True
                input_ids = d[0].to(device)
                attention_mask = d[1].to(device)
                targets_ = d[2].to(device)
                outputs_ = model(input_ids = input_ids, attention_mask = attention_mask, graph_embeddings = discovered_graph_embeddings)
                _, preds_ = torch.max(outputs_, dim = 0)
                intent_graph = graph_updater(preds_, intent_graph, last_node=prev_node, flag = flag)
                prev_node = preds_
                discovered_graph = graph_updater(targets_, discovered_graph, last_node=real_prev_node, flag=flag)
                real_prev_node = targets_
                discovered_graph_embeddings = model(graph = discovered_graph, use_graph_only = True)
                del disc_edge_list
                del intent_edge_list
                disc_edge_list = [discovered_graph.edge_index]
                intent_edge_list = [intent_graph.edge_index]
                output.append(outputs_)
                pred.append(preds_)
                target.append(targets_)

                del input_ids
                del attention_mask
                del targets_
                torch.cuda.empty_cache()
            outputs = torch.stack(output).squeeze()
            targets = torch.stack(target).squeeze()
            preds = torch.stack(pred).squeeze()
            loss = loss_CE(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            f1_values = f1_score(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), average = 'macro')
            ce_value = cascading_evaluation(targets, preds)
            em_value = exact_match_metric(targets, preds)
            f1s.append(f1_values)
            ce.append(ce_value)
            em.append(em_value)
            count += len(preds)
            losses.append(loss.item())
            optimizer_1.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)
            optimizer_1.step()

        d_g_f = Data(x = deepcopy(discovered_graph.x.detach()), edge_index= deepcopy(discovered_graph.edge_index.detach()))
        i_g_f = Data(x = deepcopy(intent_graph.x.detach()), edge_index= deepcopy(intent_graph.edge_index.detach()))

        discovered_embedds = model(graph = d_g_f, use_graph_only = True)
        intent_embedds = model(graph = i_g_f, use_graph_only = True)

        loss_2 = loss_MSE(intent_embedds, discovered_embedds)
        optimizer_1.zero_grad()
        loss_2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer_1.step()
    return correct_predictions.double() / count, np.mean(losses), np.mean(f1s), np.mean(ce), np.mean(em)


def val_epoch(model, train_data, action_label_map, loss_CE, loss_MSE, optimizer_1, device, tokenizer, semantic_mapping, bert_model, epoch):
    losses = []
    count  = 0
    correct_predictions = 0
    f1s = []
    ce = []
    em = []
    for intent_name in train_data:
        intent = train_data[intent_name]
        zero = torch.tensor([[0,0]])
        disc_edge_list = [zero.t().contiguous()]
        intent_edge_list = [zero.t().contiguous()]
        for convo in tqdm(intent['conversations']):
            action_embedding = bert_action_embedding(action_label_map, semantic_mapping, tokenizer, model=bert_model)
            d_base_edge_index = disc_edge_list[-1]
            i_base_edge_index = intent_edge_list[-1]
            intent_graph = Data(x = action_embedding.clone(), edge_index= i_base_edge_index).to(device)
            discovered_graph = Data(x=action_embedding, edge_index = d_base_edge_index).to(device)
            discovered_graph_embeddings = model(graph = discovered_graph, use_graph_only = True)
            linear_flow = None
            conversations = intent['conversations'][convo]
            clusters = context_cluster(conversations, action_label_map, 3, tokenizer)
            output = []
            pred = []
            target = []
            prev_node = 0
            real_prev_node = 0
            for index, d in enumerate(clusters):
                if index != 0:
                    flag = False
                else:
                    flag = True
                input_ids = d[0].to(device)
                attention_mask = d[1].to(device)
                targets_ = d[2].to(device)
                outputs_ = model(input_ids = input_ids, attention_mask = attention_mask, graph_embeddings = discovered_graph_embeddings)
                _, preds_ = torch.max(outputs_, dim = 0)
                intent_graph = graph_updater(preds_, intent_graph, last_node=prev_node, flag = flag)
                prev_node = preds_
                discovered_graph = graph_updater(targets_, discovered_graph, last_node=real_prev_node, flag=flag)
                real_prev_node = targets_
                discovered_graph_embeddings = model(graph = discovered_graph, use_graph_only = True)
                del disc_edge_list
                del intent_edge_list
                disc_edge_list = [discovered_graph.edge_index]
                intent_edge_list = [intent_graph.edge_index]
                output.append(outputs_)
                pred.append(preds_)
                target.append(targets_)

                del input_ids
                del attention_mask
                del targets_
                torch.cuda.empty_cache()

            #outputs = torch.stack(output).squeeze()
            targets = torch.stack(target).squeeze()
            preds = torch.stack(pred).squeeze()
            #loss = loss_CE(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            f1_values = f1_score(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), average = 'macro')
            f1s.append(f1_values)
            ce_value = cascading_evaluation(targets, preds)
            em_value = exact_match_metric(targets, preds)
            ce.append(ce_value)
            em.append(em_value)
            count += len(preds)
            #losses.append(loss.item())
            #optimizer_1.zero_grad()
            #loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)
            #optimizer_1.step()

        d_g_f = Data(x = deepcopy(discovered_graph.x.detach()), edge_index= deepcopy(discovered_graph.edge_index.detach()))
        i_g_f = Data(x = deepcopy(intent_graph.x.detach()), edge_index= deepcopy(intent_graph.edge_index.detach()))

        discovered_embedds = model(graph = d_g_f, use_graph_only = True)
        intent_embedds = model(graph = i_g_f, use_graph_only = True)

#        loss_2 = loss_MSE(intent_embedds, discovered_embedds)
#        optimizer_1.zero_grad()
#        loss_2.backward()
#        nn.utilis.clip_grad_norm_(model.parameters(), max_norm = 1.0)
#        optimizer_1.step()
    return correct_predictions.double() / count, np.mean(losses), np.mean(f1s), np.mean(ce), np.mean(em)
