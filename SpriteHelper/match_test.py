from collections import namedtuple
from logging import disable
from math import ceil, floor
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
load_dotenv()

def one_hot_encode(val, max_val=6):
    hot = [0] * max_val
    # We assume values are NOT zero indexing anything
    hot[int(val)-1] = 1
    return hot

def get_ids(char_name_array, conn):
    id_query = 'SELECT ID FROM Characters WHERE Name = ?'
    c = conn.cursor()
    went_bad = False
    for idx in range(len(char_name_array)):
        name = char_name_array[idx]
        if name is None:
            char_name_array[idx] = 0
            continue
        c.execute(id_query, (name,))
        fetched = c.fetchone()
        if fetched is None:
            print('Failed at {}'.format(name))
            went_bad = True
            break
        char_name_array[idx] = fetched[0]
    return went_bad

def parse_teams(blue_str, red_str, conn):
    blue_turns = 0
    if ' / ' in blue_str:
        blue_arr = blue_str.split(' / ')
    elif ' ⇒ ' in blue_str:
        blue_arr = blue_str.split(' ⇒ ')
        blue_turns = 1
    else:
        blue_arr = [blue_str]
    blue_arr += [None] * (4 - len(blue_arr))
    blue_raw = [x for x in blue_arr]
    
    red_turns = 0
    if ' / ' in red_str:
        red_arr = red_str.split(' / ')
    elif ' ⇒ ' in red_str:
        red_arr = red_str.split(' ⇒ ')
        red_turns = 1
    else:
        red_arr = [red_str]
    red_arr += [None] * (4 - len(red_arr))
    red_raw = [x for x in red_arr]

    blue_went_bad = get_ids(blue_arr, conn)
    red_went_bad = get_ids(red_arr, conn)

    if blue_went_bad or red_went_bad:
        return None

    total_arr = blue_arr + [blue_turns] + red_arr + [red_turns]
    return total_arr, blue_raw, red_raw

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

def closest_point(point, pointList):
    nodes = np.asarray(pointList)
    dist_2 = np.sum((nodes - point)**2, axis=1)
    return np.argmin(dist_2)


class SpriteHelper(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SpriteHelper, self).__init__(parent)
        self.lmodel_button = QtWidgets.QPushButton('Load Model')
        self.lmodel_label = QtWidgets.QLabel('None')
        self.ldb_button = QtWidgets.QPushButton('Load DB')
        self.ldb_label = QtWidgets.QLabel('None')
        self.connect_button = QtWidgets.QPushButton('Connect')
        self.connect_label = QtWidgets.QLabel('Disconnected')
        self.disconnect_button = QtWidgets.QPushButton('Disconnect')
        self.model_filename = None
        self.db_filename = None

        large_layout = QtWidgets.QVBoxLayout()
        
        load_layout = QtWidgets.QGridLayout()
        load_layout.addWidget(self.lmodel_button,0,0)
        load_layout.addWidget(self.lmodel_label,0,1)
        load_layout.addWidget(self.ldb_button,1,0)
        load_layout.addWidget(self.ldb_label,1,1)
        load_layout.addWidget(self.connect_button,2,0)
        load_layout.addWidget(self.connect_label,2,1)
        load_layout.addWidget(self.disconnect_button,3,0)

        self.blue_0_label = QtWidgets.QLabel('Blue Team')
        self.blue_0_textbox = QtWidgets.QLineEdit('')

        self.red_0_label = QtWidgets.QLabel('Red Team')
        self.red_0_textbox = QtWidgets.QLineEdit('')

        self.estimate_button = QtWidgets.QPushButton('Estimate odds')

        
        info_layout = QtWidgets.QGridLayout()

        info_layout.addWidget(self.blue_0_label,0,0, alignment=QtCore.Qt.AlignCenter)
        info_layout.addWidget(self.blue_0_textbox,1,0)
        info_layout.addWidget(self.red_0_label,0,1, alignment=QtCore.Qt.AlignCenter)
        info_layout.addWidget(self.red_0_textbox,1,1)
        info_layout.addWidget(self.estimate_button,2,0,1,4)

        pred_layout = QtWidgets.QGridLayout()

        self.pred_label = QtWidgets.QLabel('Predictions')
        self.pred_blue_label = QtWidgets.QLabel('Blue')
        self.pred_red_label = QtWidgets.QLabel('Red')
        self.pred_bad_label = QtWidgets.QLabel('Bad')
        self.blue_pred_textbox = QtWidgets.QLineEdit('')
        self.blue_pred_textbox.setReadOnly(True)
        self.red_pred_textbox = QtWidgets.QLineEdit('')
        self.red_pred_textbox.setReadOnly(True)
        self.bad_pred_textbox = QtWidgets.QLineEdit('')
        self.bad_pred_textbox.setReadOnly(True)
        self.emb_plot = pg.PlotWidget()
        self.emb_plot.setXRange(-100, 100, padding=0)
        self.emb_plot.setYRange(-100, 100, padding=0)
        self.sel_char_label = QtWidgets.QLabel('Selected character')
        self.sel_char_textbox = QtWidgets.QLineEdit('')
        self.sel_char_textbox.setReadOnly(True)
        self.sel_emb_label = QtWidgets.QLabel('Embeddings')
        self.sel_emb_table = QtWidgets.QTableWidget()

        pred_layout.addWidget(self.pred_label,0,0,1,3, alignment=QtCore.Qt.AlignCenter)
        pred_layout.addWidget(self.pred_blue_label,1,0)
        pred_layout.addWidget(self.pred_red_label,1,1)
        pred_layout.addWidget(self.pred_bad_label,1,2)
        pred_layout.addWidget(self.blue_pred_textbox,2,0)
        pred_layout.addWidget(self.red_pred_textbox,2,1)
        pred_layout.addWidget(self.bad_pred_textbox,2,2)
        pred_layout.addWidget(self.emb_plot,3,0,2,3)
        pred_layout.addWidget(self.sel_char_label,5,0,1,1)
        pred_layout.addWidget(self.sel_char_textbox,5,1,1,3)
        pred_layout.addWidget(self.sel_emb_label,6,0,1,1)
        pred_layout.addWidget(self.sel_emb_table,6,1,1,3)



        large_layout.addLayout(load_layout)
        large_layout.addLayout(info_layout)
        large_layout.addLayout(pred_layout)

        self.setLayout(large_layout)
        self.lmodel_button.clicked.connect(self.load_model)
        self.ldb_button.clicked.connect(self.load_db)
        self.connect_button.clicked.connect(self.try_connect)
        self.estimate_button.clicked.connect(self.try_estimate)

        self.db = None
        self.model = None
        self.model_filename = ''
        self.db_filename = ''

    
    def load_model(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Model', '..', self.tr('Model files (*.hdf5 *.h5)'))
        if filename[0] != '':
            self.model_filename = filename[0]
            self.lmodel_label.setText(filename[0])


    def load_db(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Model', '..', 'Database files (*.db)')
        if filename[0] != '':
            self.db_filename = filename[0]
            self.ldb_label.setText(filename[0])
    
    def try_connect(self):
        bad = False
        if self.model_filename is None:
            self.lmodel_label.setText('<font color=red>Please select a model first</font>')
            bad = True
        if self.db_filename is None:
            self.ldb_label.setText('<font color=red>Please select a database first</font>')
            bad = True
        if not bad:
            self.model = tf.keras.models.load_model(self.model_filename, custom_objects={'SpecialEmbedding': SpecialEmbedding})
            self.db = sqlite3.connect(self.db_filename)
            self.embeddings = get_all_embeddings(self.db, self.model)
            if os.path.exists('precomputed_tsne.npy'):
                self.coordinates = np.load('precomputed_tsne.npy')
            else:
                self.coordinates = TSNE(n_components=2).fit_transform(list(self.embeddings.values()))
            comp = QtWidgets.QCompleter(list(self.embeddings.keys()))
            comp.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            self.blue_0_textbox.setCompleter(comp)
            self.red_0_textbox.setCompleter(comp)
            self.connect_label.setText('<font color=green>Connected</font>')
    
    def try_estimate(self):
        blue = self.blue_0_textbox.text()
        red = self.red_0_textbox.text()
        blue_array = [blue, None, None, None]
        red_array = [red, None, None, None]
        blue_bad = get_ids(blue_array, self.db)
        red_bad = get_ids(red_array, self.db)
        if blue_bad:
            self.blue_0_textbox.clear()
        if red_bad is None:
            self.red_0_textbox.clear()
        if red_bad or blue_bad:
            return
        total_match = np.asarray([blue_array+red_array])
        preds = self.model(total_match).numpy()
        red_win = preds[0][0]
        blue_win = 1 - red_win
        self.blue_pred_textbox.setText(str(blue_win))
        self.red_pred_textbox.setText(str(red_win))
        

        
        

if __name__ == '__main__':
    # Create the Qt Application
    app = QtWidgets.QApplication(sys.argv)
    # Create and show the form
    form = SpriteHelper()
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())