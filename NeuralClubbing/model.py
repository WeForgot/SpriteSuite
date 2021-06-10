import json
import os
import random
import sqlite3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def get_parameter_count(model: nn.Module):
    t_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    u_model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    t_params = sum([np.prod(p.size()) for p in t_model_parameters])
    u_params = sum([np.prod(p.size()) for p in u_model_parameters])
    return t_params, u_params

def get_data(sqlite_path, include_exhibs=False, include_odd_outcomes=False):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    features = []
    labels = []
    num_characters = cur.execute('SELECT COUNT(*) FROM Characters').fetchone()[0]
    all_matches = cur.execute('SELECT Blue_3, Blue_2, Blue_1, Blue_0, Red_0, Red_1, Red_2, Red_3, Blue_Turns, Red_Turns, Outcome, Session FROM Matches').fetchall()
    for x in all_matches:
        x = list(x)
        if x[-1] == 'Exhibitions' and not include_exhibs:
            continue
        if x[-2] > 1 and not include_odd_outcomes:
            continue
        vs_type_token = 0
        blue_turns, red_turns = x[-4], x[-3]
        # While extreme rare, it is possible to have turns vs non-turns
        if blue_turns and not red_turns:
            vs_type_token = 2
        elif red_turns and not blue_turns:
            vs_type_token = 3
        elif blue_turns and red_turns:
            vs_type_token = 4
        else:
            vs_type_token = 1
        contestants = x[:8]
        contestants.insert(4, vs_type_token)
        contestants = [x+4 for x in contestants if x is not None]
        features.append(torch.tensor(contestants))
        labels.append(torch.tensor(x[-2]))
    return num_characters, features, labels

class SpriteDataset(Dataset):
    def __init__(self, features, labels, many_class=False):
        assert len(features) == len(labels), 'Length of features and labels must be equal'
        self.features = features
        self.labels = labels
        self.many_class = many_class
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.as_tensor(self.features[idx]).long(), torch.as_tensor(self.labels[idx]).long()

class SpriteModel(nn.Module):
    def __init__(self, vocab_size, emb_size, num_classes, hidden_size=16, depth=2):
        super(SpriteModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=depth, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(normalized_shape=hidden_size)
        self.to_logits = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.depth = depth
        self.hidden_size = hidden_size
    
    def forward(self, x_padded, x_lens):
        x = self.embedding(x_padded)
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(x) # Get hidden states
        batch_size = h.shape[1]
        h = h.view(self.depth, 2, batch_size, self.hidden_size)[-1] # Only at the last layer
        x = torch.mean(h, dim=0) # Because the leading dimension is the hidden states at final timestep for each direction, we reduce them
        x = self.to_logits(x)
        return x

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy = torch.as_tensor(yy)
    return xx_pad, yy, x_lens

def main():
    debug = False # Debugging your model on CPU is leagues easier
    if debug:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')

    r_seed = 420
    random.seed(r_seed)
    torch.manual_seed(r_seed)
    use_odd_outcomes = False
    num_labels = 3 if use_odd_outcomes else 2
    num_characters, features, labels = get_data(os.path.join('..', 'MatchScraper', 'sprite.db'))
    temp = list(zip(features, labels))
    random.shuffle(temp)
    features, labels = zip(*temp)

    batch_size = 512
    valid_split = 0.2
    valid_start = int(len(features) - (len(features) * valid_split))

    train_dataset, valid_dataset = SpriteDataset(features[:valid_start], labels[:valid_start]), SpriteDataset(features[valid_start:], labels[valid_start:])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=pad_collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=pad_collate)

    model = SpriteModel(num_characters+5, 4, num_labels).to(device)
    with open('meta.json', 'w') as f:
        json.dump({'vocab_size': num_characters+5, 'emb_size': 4, 'num_labels': num_labels}, f)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    print('Total model parameters:\n\tTrainable: {}\n\tUntrainable: {}'.format(*(get_parameter_count(model))))

    best_loss = None
    best_model = None
    patience = 0
    max_patience = 100
    show_batch_metrics = False
    
    epochs = 1000000000
    print('Begginning training')
    for edx in range(epochs):
        model.train()
        running_loss = 0.0
        for bdx, (x_padded, y, x_lens) in enumerate(train_dataloader):
            x_padded, y, x_lens = x_padded.to(device), y.to(device), x_lens
            opt.zero_grad()
            x = model(x_padded, x_lens)
            loss = F.cross_entropy(x, y)
            running_loss += loss.item()
            loss.backward()
            opt.step()
            if show_batch_metrics:
                print('\tBatch #{}, Loss: {}'.format(bdx, loss.item()))
        print('TRAINING: Epoch #{}, Loss: {}'.format(edx, running_loss))

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for bdx, (x_padded, y, x_lens) in enumerate(valid_dataloader):
                x_padded, y, x_lens = x_padded.to(device), y.to(device), x_lens
                x = model(x_padded, x_lens)
                loss = F.cross_entropy(x, y, reduction='sum')
                running_loss += loss.item()
        
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            print('Out of patience')
            break
        print('VALIDATION: Epoch #{}, Loss: {}, Average: {}, Patience: {}/{}'.format(edx, running_loss, running_loss/batch_size/len(valid_dataloader), patience, max_patience))

        
    
    torch.save(best_model, 'best_model.pt')



if __name__ == '__main__':
    main()