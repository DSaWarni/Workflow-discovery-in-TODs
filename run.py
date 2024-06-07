import torch
import json
from sklearn.metrics import f1_score
from build_multiplex_network.base_file import *
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import transformers
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
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


path = 'dataset.json'

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

path2 = 'abcd_action_mappings.json'

with open(path2, 'r', encoding='utf-8') as f:
    semantic_mapping = json.load(f)

action_label_map = {i: index+1 for index, i in enumerate(semantic_mapping.keys())}
action_label_map[None] = 0
semantic_mapping[None] = 'No Action taken'

model = Model_base()
model.to(device)
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)

train_data = {}
val_data = {}

train_intents = ['manage_upgrade']

val_intents = ["manage"]

for i in train_intents:
    train_data[i] = data[i]

for i in val_intents:
    val_data[i] = data[i]


EPOCHS = 10

optimizer_1  = AdamW(model.parameters(), lr = 2e-5, correct_bias = False)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
loss_CE = nn.CrossEntropyLoss()
loss_MSE = nn.MSELoss()

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch+1}/{EPOCHS}')
    print('-'*10)

    train_acc, train_loss, train_f1s, train_ce, train_em = train_epoch(model, train_data, action_label_map,
                                                   loss_CE,
                                                   loss_MSE,
                                                   optimizer_1,
                                                   device,
                                                   tokenizer,
                                                   semantic_mapping,
                                                   bert_model,
                                                   epoch)
    print(f'Train loss {train_loss} accuracy {train_acc} f1 {train_f1s} ce {train_ce} em {train_em}')

    val_acc, val_loss, val_f1s, val_ce, val_em = val_epoch(model, val_data, action_label_map,
                                           loss_CE,
                                           loss_MSE,
                                           optimizer_1,
                                           device,
                                           tokenizer,
                                           semantic_mapping,
                                           bert_model,
                                           epoch)
    print(f'Val Loss {val_loss}  accuracy {val_acc} f1 {val_f1s} ce {val_ce} em {val_em}')
