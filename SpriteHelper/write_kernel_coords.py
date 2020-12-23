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
    model = tf.keras.models.load_model('vl_0.662_ca_0.674_8_113.hdf5', custom_objects={'SpecialEmbedding': SpecialEmbedding})
    db = sqlite3.connect('sprite.db')
    embeddings = list(get_all_embeddings(db, model).values())
    kpca = KernelPCA(n_components=2).fit_transform(embeddings[1:])
    kpca = np.insert(kpca, 0, np.zeros(2,), axis=0)
    print(kpca)
    print(np.max(kpca))
    print(np.min(kpca))
    np.save('pca_reductions.npy', kpca)


if __name__ == '__main__':
    main()