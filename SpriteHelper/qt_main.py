from collections import namedtuple
from logging import disable
import os
import re
import socket
import sqlite3
import sys
import time

import numpy as np
from dotenv import load_dotenv
from PySide6 import QtWidgets, QtCore, QtGui
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

class SpriteHelper(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SpriteHelper, self).__init__(parent)
        self.lmodel_button = QtWidgets.QPushButton('Load Model')
        self.lmodel_label = QtWidgets.QLabel('None')
        self.ldb_button = QtWidgets.QPushButton('Load DB')
        self.ldb_label = QtWidgets.QLabel('None')
        self.connect_button = QtWidgets.QPushButton('Connect')
        self.disconnect_button = QtWidgets.QPushButton('Disconnect')
        self.model_filename = None
        self.db_filename = None

        large_layout = QtWidgets.QGridLayout()

        large_layout.addWidget(self.lmodel_button,0,0)
        large_layout.addWidget(self.lmodel_label,0,1)
        large_layout.addWidget(self.ldb_button,1,0)
        large_layout.addWidget(self.ldb_label,1,1)
        large_layout.addWidget(self.connect_button,2,0)
        large_layout.addWidget(self.disconnect_button,3,0)

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

        self.pred_label = QtWidgets.QLabel('Predictions')
        self.blue_pred_textbox = QtWidgets.QLineEdit('')
        self.blue_pred_textbox.setReadOnly(True)
        self.red_pred_textbox = QtWidgets.QLineEdit('')
        self.red_pred_textbox.setReadOnly(True)
        

        large_layout.addWidget(self.blue_0_label,4,0, alignment=QtCore.Qt.AlignCenter)
        large_layout.addWidget(self.blue_0_textbox,5,0)
        large_layout.addWidget(self.blue_1_textbox,6,0)
        large_layout.addWidget(self.blue_2_textbox,7,0)
        large_layout.addWidget(self.blue_3_textbox,8,0)
        large_layout.addWidget(self.red_0_label,4,1, alignment=QtCore.Qt.AlignCenter)
        large_layout.addWidget(self.red_0_textbox,5,1)
        large_layout.addWidget(self.red_1_textbox,6,1)
        large_layout.addWidget(self.red_2_textbox,7,1)
        large_layout.addWidget(self.red_3_textbox,8,1)
        large_layout.addWidget(self.)

        self.setLayout(large_layout)
        self.lmodel_button.clicked.connect(self.load_model)
        self.ldb_button.clicked.connect(self.load_db)
        self.connect_button.clicked.connect(self.try_connect)
        self.disconnect_button.clicked.connect(self.try_disconnect)
    
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
    
    def signal_recieved(self, message):
        self.toggle_edit(True)
        self.blue_0_textbox.setText(message.blue_0) if message.blue_0 is not None else ''
        self.blue_1_textbox.setText(message.blue_1) if message.blue_1 is not None else ''
        self.blue_2_textbox.setText(message.blue_2) if message.blue_2 is not None else ''
        self.blue_3_textbox.setText(message.blue_3) if message.blue_3 is not None else ''

        self.red_0_textbox.setText(message.red_0) if message.red_0 is not None else ''
        self.red_1_textbox.setText(message.red_1) if message.red_1 is not None else ''
        self.red_2_textbox.setText(message.red_2) if message.red_2 is not None else ''
        self.red_3_textbox.setText(message.red_3) if message.red_3 is not None else ''
        self.toggle_edit(False)
    
    def try_disconnect(self):
        self.irc_listener.running = False
        self.thread.terminate()

BettingInformation = namedtuple('BettingInformation', ['blue_raw', 'blue_0', 'blue_1', 'blue_2', 'blue_3',
                                                       'blue_0_e', 'blue_1_e', 'blue_2_e', 'blue_3_e',
                                                       'red_raw', 'red_0', 'red_1', 'red_2', 'red_3', 
                                                       'red_0_e', 'red_1_e', 'red_2_e', 'red_3_e',
                                                       'blue_prob', 'red_prob', 'bad_prob'])
class IRC_Listener(QtCore.QObject):
    message = QtCore.Signal(BettingInformation)

    def __init__(self, model_path, db_path):
        QtCore.QObject.__init__(self)
        self.model_path = model_path
        self.db_path = db_path
        

        self.running = True
    
    def loop(self):
        model = tf.keras.models.load_model(self.model_path, custom_objects={'SpecialEmbedding': SpecialEmbedding})
        db = sqlite3.connect(self.db_path)
        embeddings = get_all_embeddings(db, model)
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
                                                              raw_red, split_red[0], split_red[1], split_red[2], split_red[3],
                                                              embeddings[split_red[0]], embeddings[split_red[1]], embeddings[split_red[2]], embeddings[split_red[3]],
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