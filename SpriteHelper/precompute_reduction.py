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
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.decomposition import KernelPCA
import tensorflow as tf

sys.path.append('..')
from NeuralClubbing.model import SpecialEmbedding

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
    model = tf.keras.models.load_model('..\\NeuralClubbing\\checkpoints\\binary_vl_0.579_ca_0.690_9_64-128.hdf5', custom_objects={'SpecialEmbedding': SpecialEmbedding})
    db = sqlite3.connect('..\\MatchScraper\\sprite.db')
    emb_dict = get_all_embeddings(db, model)
    embeddings = list(emb_dict.values())
    coordinates = TSNE(n_components=2).fit_transform(embeddings[1:])
    coordinates = np.insert(coordinates, 0, np.zeros(2,), axis=0)
    np.save('precomputed_tsne.npy', coordinates)
    with open('characters.tsv', 'w', encoding='utf-8') as cf, open('weights.tsv', 'w') as wf:
        for emb_name in emb_dict:
            cf.write('{}\n'.format(emb_name))
            wf.write('\t'.join(list(map(str, emb_dict[emb_name]))) + '\n')


if __name__ == '__main__':
    main()