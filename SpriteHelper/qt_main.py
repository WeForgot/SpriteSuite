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
load_dotenv()

debug_line = 'Betting open for: [ Caviar? ⇒ Presea Combatir ⇒ Burai Yamamoto ⇒ Rinnosuke Morichika ] Vs. [ Ryuken ⇒ SS6 Senna ⇒ Nogami Neuro ⇒ Jedah K ] (5th Division matchmaking)'

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
        self.blue_0_textbox.setReadOnly(True)
        self.blue_1_textbox = QtWidgets.QLineEdit('')
        self.blue_1_textbox.setReadOnly(True)
        self.blue_2_textbox = QtWidgets.QLineEdit('')
        self.blue_2_textbox.setReadOnly(True)
        self.blue_3_textbox = QtWidgets.QLineEdit('')
        self.blue_3_textbox.setReadOnly(True)

        self.red_0_label = QtWidgets.QLabel('Red Team')
        self.red_0_textbox = QtWidgets.QLineEdit('')
        self.red_0_textbox.setReadOnly(True)
        self.red_1_textbox = QtWidgets.QLineEdit('')
        self.red_1_textbox.setReadOnly(True)
        self.red_2_textbox = QtWidgets.QLineEdit('')
        self.red_2_textbox.setReadOnly(True)
        self.red_3_textbox = QtWidgets.QLineEdit('')
        self.red_3_textbox.setReadOnly(True)

        
        info_layout = QtWidgets.QGridLayout()

        info_layout.addWidget(self.blue_0_label,0,0, alignment=QtCore.Qt.AlignCenter)
        info_layout.addWidget(self.blue_0_textbox,1,0)
        info_layout.addWidget(self.blue_1_textbox,2,0)
        info_layout.addWidget(self.blue_2_textbox,3,0)
        info_layout.addWidget(self.blue_3_textbox,4,0)
        info_layout.addWidget(self.red_0_label,0,1, alignment=QtCore.Qt.AlignCenter)
        info_layout.addWidget(self.red_0_textbox,1,1)
        info_layout.addWidget(self.red_1_textbox,2,1)
        info_layout.addWidget(self.red_2_textbox,3,1)
        info_layout.addWidget(self.red_3_textbox,4,1)

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
        self.emb_plot.setXRange(-1, 1, padding=0)
        self.emb_plot.setYRange(-1, 1, padding=0)
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
        self.disconnect_button.clicked.connect(self.try_disconnect)

        self.cbi = None # Current Betting Information
        self.point_to_name = {}
        self.emb_scatter = None
    
    def load_model(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Model', '.', self.tr('Model files (*.hdf5 *.h5)'))
        if filename[0] != '':
            self.model_filename = filename[0]
            self.lmodel_label.setText(filename[0])


    def load_db(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Model', '.', 'Database files (*.db)')
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
            self.thread = QtCore.QThread()
            self.irc_listener = IRC_Listener(self.model_filename, self.db_filename)
            self.irc_listener.moveToThread(self.thread)
            self.thread.started.connect(self.irc_listener.loop)
            self.irc_listener.message.connect(self.signal_recieved)
            self.irc_listener.readying.connect(self.update_status)
            QtCore.QTimer.singleShot(0, self.thread.start)
    
    def toggle_edit(self, editable):
        self.blue_0_textbox.setReadOnly(not editable)
        self.blue_1_textbox.setReadOnly(not editable)
        self.blue_2_textbox.setReadOnly(not editable)
        self.blue_3_textbox.setReadOnly(not editable)

        self.red_0_textbox.setReadOnly(not editable)
        self.red_1_textbox.setReadOnly(not editable)
        self.red_2_textbox.setReadOnly(not editable)
        self.red_3_textbox.setReadOnly(not editable)
        
        self.blue_pred_textbox.setReadOnly(not editable)
        self.red_pred_textbox.setReadOnly(not editable)
        self.bad_pred_textbox.setReadOnly(not editable)

        self.sel_char_textbox.setReadOnly(not editable)


    def update_betting_ui(self):
        if self.cbi is None:
            return
        self.toggle_edit(True)
        self.blue_0_textbox.setText(self.cbi.blue_0) if self.cbi.blue_0 is not None else ''
        self.blue_1_textbox.setText(self.cbi.blue_1) if self.cbi.blue_1 is not None else ''
        self.blue_2_textbox.setText(self.cbi.blue_2) if self.cbi.blue_2 is not None else ''
        self.blue_3_textbox.setText(self.cbi.blue_3) if self.cbi.blue_3 is not None else ''

        self.red_0_textbox.setText(self.cbi.red_0) if self.cbi.red_0 is not None else ''
        self.red_1_textbox.setText(self.cbi.red_1) if self.cbi.red_1 is not None else ''
        self.red_2_textbox.setText(self.cbi.red_2) if self.cbi.red_2 is not None else ''
        self.red_3_textbox.setText(self.cbi.red_3) if self.cbi.red_3 is not None else ''

        self.blue_pred_textbox.setText('{:.3f}'.format(self.cbi.blue_prob))
        self.red_pred_textbox.setText('{:.3f}'.format(self.cbi.red_prob))
        self.bad_pred_textbox.setText('{:.3f}'.format(self.cbi.bad_prob))
        self.toggle_edit(False)

        names = []
        blue_points = []
        red_points = []
        reduced_embs = []
        if self.cbi.blue_0 is not None:
            blue_points.append(self.cbi.blue_0_e)
            reduced_embs.append(self.cbi.blue_0_re)
            names.append(self.cbi.blue_0)
        if self.cbi.blue_1 is not None:
            blue_points.append(self.cbi.blue_1_e)
            reduced_embs.append(self.cbi.blue_1_re)
            names.append(self.cbi.blue_1)
        if self.cbi.blue_2 is not None:
            blue_points.append(self.cbi.blue_2_e)
            reduced_embs.append(self.cbi.blue_2_re)
            names.append(self.cbi.blue_2)
        if self.cbi.blue_3 is not None:
            blue_points.append(self.cbi.blue_3_e)
            reduced_embs.append(self.cbi.blue_3_re)
            names.append(self.cbi.blue_3)
        if self.cbi.red_0 is not None:
            red_points.append(self.cbi.red_0_e)
            reduced_embs.append(self.cbi.red_0_re)
            names.append(self.cbi.red_0)
        if self.cbi.red_1 is not None:
            red_points.append(self.cbi.red_1_e)
            reduced_embs.append(self.cbi.red_1_re)
            names.append(self.cbi.red_1)
        if self.cbi.red_2 is not None:
            red_points.append(self.cbi.red_2_e)
            reduced_embs.append(self.cbi.red_2_re)
            names.append(self.cbi.red_2)
        if self.cbi.red_3 is not None:
            red_points.append(self.cbi.red_3_e)
            reduced_embs.append(self.cbi.red_2_re)
            names.append(self.cbi.red_3)
        total_points = np.asarray(blue_points + red_points)
            
        blue_brush = pg.mkBrush((0,0,255))
        red_brush = pg.mkBrush((255,0,0))
        blue_x = [x[0] for x in reduced_embs[:len(blue_points)]]
        blue_y = [x[1] for x in reduced_embs[:len(blue_points)]]
        red_x = [x[0] for x in reduced_embs[len(blue_points):]]
        red_y = [x[1] for x in reduced_embs[len(blue_points):]]
        total_x = blue_x + red_x
        total_y = blue_y + red_y
        total_brush = [blue_brush] * len(blue_points) + [red_brush] * len(red_points)
        self.emb_plot.clear()
        self.emb_scatter = self.emb_plot.plot(x=total_x, y=total_y, pen=None, symbol='o', symbolBrush=total_brush, hoverable=True, hoverSymbol='s')

        def set_selected(ev, point):
            idx = closest_point(np.asarray(point.pos()), reduced_embs)
            closest_char = names[idx]
            self.toggle_edit(True)
            self.sel_char_textbox.setText(closest_char)
            self.toggle_edit(False)
            self.sel_emb_table.setRowCount(1)
            self.sel_emb_table.setColumnCount(len(total_points[idx]))
            for x in range(len(total_points[idx])):
                self.sel_emb_table.setItem(0, x, QtWidgets.QTableWidgetItem('{:.5f}'.format(total_points[idx][x])))

        self.emb_scatter.sigClicked.connect(set_selected)
    
    def signal_recieved(self, message):
        self.cbi = message
        if message is not None:
            self.update_betting_ui()
    
    def try_disconnect(self):
        self.irc_listener.running = False
        self.thread.terminate()
        self.connect_label.setText('Disconnected')
    
    def update_status(self, message):
        self.connect_label.setText(message)


BettingInformation = namedtuple('BettingInformation', ['blue_raw', 'blue_0', 'blue_1', 'blue_2', 'blue_3',
                                                       'blue_0_e', 'blue_1_e', 'blue_2_e', 'blue_3_e',
                                                       'blue_0_re', 'blue_1_re', 'blue_2_re', 'blue_3_re',
                                                       'red_raw', 'red_0', 'red_1', 'red_2', 'red_3', 
                                                       'red_0_e', 'red_1_e', 'red_2_e', 'red_3_e',
                                                       'red_0_re', 'red_1_re', 'red_2_re', 'red_3_re',
                                                       'blue_prob', 'red_prob', 'bad_prob'])
class IRC_Listener(QtCore.QObject):
    message = QtCore.Signal(BettingInformation)
    readying = QtCore.Signal(str)

    def __init__(self, model_path, db_path):
        QtCore.QObject.__init__(self)
        self.model_path = model_path
        self.db_path = db_path
        

        self.running = True
    
    def loop(self):
        self.readying.emit('Loading model and DB')
        model = tf.keras.models.load_model(self.model_path, custom_objects={'SpecialEmbedding': SpecialEmbedding})
        db = sqlite3.connect(self.db_path)
        self.readying.emit('Collecting embeddings')
        embeddings = get_all_embeddings(db, model)
        self.readying.emit('Training KernelPCA')
        kpca = KernelPCA(n_components=2).fit(list(embeddings.values())[1:])
        self.readying.emit('Connecting to IRC')
        server = 'irc.chat.twitch.tv'
        port = 6667
        nickname = os.getenv('NICKNAME')
        token = os.getenv('TOKEN')
        channel = '#spriteclub'
        sock = socket.socket()
        sock.connect((server,port))
        sock.settimeout(360)
        sock.send(f'PASS {token}\n'.encode('utf-8'))
        sock.send(f'NICK {nickname}\n'.encode('utf-8'))
        sock.send(f'JOIN {channel}\n'.encode('utf-8'))
        self.readying.emit('<font color=green>Connected</font>')
        
        while self.running:
            msg = sock.recv(2048).decode('utf-8')
            if msg.startswith('PING'):
                sock.send('PONG\n'.encode('utf-8'))
            elif len(msg) > 0:
                if 'PRIVMSG' in msg:
                    username, channel, message = re.search(':(.*)\!.*@.*\.tmi\.twitch\.tv PRIVMSG #(.*) :(.*)', msg).groups()
                    if username == 'spriteclub':
                        if 'Betting open' in message:
                            if '(Exhibition by ' in message:
                                continue
                            to_strip_l = message.find('Betting open for: ') + len('Betting open for: ')
                            to_strip_r = message.rfind(' (')
                            message = message[to_strip_l:to_strip_r]
                            raw_blue, raw_red = message.split(' Vs. ')
                            if raw_blue.startswith('[ '):
                                raw_blue = raw_blue[2:-2]
                            if raw_red.startswith('[ '):
                                raw_red = raw_red[2:-2]
                            match_array, split_blue, split_red = parse_teams(raw_blue, raw_red, db)
                            if match_array is None:
                                continue
                            predictions = model(np.asarray([match_array]))[0].numpy()
                            cur_pred = np.argmax(predictions)
                            if cur_pred == 0:
                                tex_pred = ' blue'
                            elif cur_pred == 1:
                                tex_pred = ' red'
                            else:
                                tex_pred = ' at your own risk'
                            current_info = BettingInformation(raw_blue, split_blue[0], split_blue[1], split_blue[2], split_blue[3],
                                                              embeddings[split_blue[0]], embeddings[split_blue[1]], embeddings[split_blue[2]], embeddings[split_blue[3]],
                                                              kpca.transform([embeddings[split_blue[0]]])[0] if split_blue[0] != 0 else None, 
                                                              kpca.transform([embeddings[split_blue[1]]])[0] if split_blue[1] != 0 else None, 
                                                              kpca.transform([embeddings[split_blue[2]]])[0] if split_blue[2] != 0 else None, 
                                                              kpca.transform([embeddings[split_blue[3]]])[0] if split_blue[3] != 0 else None, 
                                                              raw_red, split_red[0], split_red[1], split_red[2], split_red[3],
                                                              embeddings[split_red[0]], embeddings[split_red[1]], embeddings[split_red[2]], embeddings[split_red[3]],
                                                              kpca.transform([embeddings[split_red[0]]])[0] if split_red[0] != 0 else None, 
                                                              kpca.transform([embeddings[split_red[1]]])[0] if split_red[1] != 0 else None, 
                                                              kpca.transform([embeddings[split_red[2]]])[0] if split_red[2] != 0 else None, 
                                                              kpca.transform([embeddings[split_red[3]]])[0] if split_red[3] != 0 else None, 
                                                              predictions[0],predictions[1],predictions[2])
                            self.message.emit(current_info)

                        elif 'Winner: ' in message:
                            pass

        print('Killed')
    


if __name__ == '__main__':
    # Create the Qt Application
    app = QtWidgets.QApplication(sys.argv)
    # Create and show the form
    form = SpriteHelper()
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())