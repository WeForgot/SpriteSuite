import csv
import os
import sqlite3

import numpy as np
import torch

def get_data(sqlite_path, include_odd_outcomes=False):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    num_characters = cur.execute('SELECT COUNT(*) FROM Characters').fetchone()[0] + 1
    all_characters = cur.execute('SELECT Name, Division FROM Characters')
    label_max = 2 if include_odd_outcomes else 3
    with open('labels.tsv', 'wt', encoding='utf-8', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['name','division'])
        for x in all_characters:
            x = [x.strip() for x in x]
            print(list(x))
            tsv_writer.writerow(list(x))

    return num_characters, label_max
        

def main():
    num_characters, labels = get_data(os.path.join('..', 'MatchScraper', 'sprite.db'))
    state_dict = torch.load('best_model.pt')
    parameters = state_dict['embedding.weight'].cpu().numpy()[5:]
    np.savetxt('embeddings.tsv', parameters, delimiter='\t')

if __name__ == '__main__':
    main()