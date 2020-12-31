from collections import namedtuple
from logging import disable
import os
import pickle
import re
import socket
import sqlite3
import sys
import time

import numpy as np
from dotenv import load_dotenv
from PySide2 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from sklearn.manifold import TSNE
import tensorflow as tf

from model import SpecialEmbedding

def get_all_embeddings(conn, model):
    emb = {}
    embeddings = np.asarray(model.layers[0].embedding.get_weights()[0])
    c = conn.cursor()
    c.execute('SELECT * FROM Characters')
    results = c.fetchall()
    emb[None] = np.zeros(shape=embeddings[0].shape)
    for result in results:
        emb[result[1]] = embeddings[result[0]]
    return emb

def main():
    model = tf.keras.models.load_model('.\\checkpoints\\binary_vl_0.569_ca_0.701_2_8-16.hdf5', custom_objects={'SpecialEmbedding': SpecialEmbedding})
    db = sqlite3.connect('..\\MatchScraper\\sprite.db')
    emb_dict = get_all_embeddings(db, model)
    embeddings = list(emb_dict.values())
    with open('characters.tsv', 'w', encoding='utf-8') as cf, open('weights.tsv', 'w') as wf, open('total_embs.tsv', 'w', encoding='utf-8') as total:
        total.write('Name\tx\ty\n')
        for emb_name in emb_dict:
            if emb_name is None:
                continue
            cf.write('{}\n'.format(emb_name))
            wf.write('\t'.join(list(map(str, emb_dict[emb_name]))) + '\n')
            total.write('{}\t{}\t{}\n'.format(emb_name, emb_dict[emb_name][0], emb_dict[emb_name][1]))
    print(np.max(embeddings))
    print(np.min(embeddings))


if __name__ == '__main__':
    main()