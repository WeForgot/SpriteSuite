from logging import disable
import os
import re
import socket
import sqlite3
import sys
import time

from dearpygui.core import *
from dearpygui.simple import *

import numpy as np
from dotenv import load_dotenv
import tensorflow as tf

sys.path.append('..')
from NeuralClubbing.model import SpecialEmbedding
load_dotenv()

Winner_Hint = {
        0: 'Blue',
        1: 'Red',
        2: 'Crash',
        3: 'Skipped',
        4: 'Tie',
        5: 'Timeout'
    }

embeddings = None
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
        # Uncomment below if there is any reason to include these
        # From what I can tell they are either patched AI or replacements of the same name
        #cleaned_name = name.replace(' (dupe)','') if name.endswith('(dupe)') else name
        #cleaned_name = name.replace(' (old)','') if name.endswith('(old)') else name
        #c.execute(id_query, (cleaned_name,))
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

def start_up(sender, data):
    if get_value('running'):
        log_error('Inference already started')
        return
    model_path = get_value('model_path')
    db_path = get_value('db_path')
    if model_path == None or model_path == '':
        set_value('start_errors', 'No model specified')
        return
    elif db_path == None or db_path == '':
        set_value('start_errors', 'No DB specified')
        return
    set_value('running', True)
    run_async_function(run_prediction_async, 'whatever', return_handler=prediction_return_handler)

def set_team_values(val_dict: dict):
    for key in val_dict:
        set_value(key, val_dict[key])

def get_all_embeddings(conn, model):
    global embeddings
    emb = {}
    embeddings = np.asarray(model.layers[0].embedding.get_weights()[0])
    c = conn.cursor()
    c.execute('SELECT * FROM Characters')
    results = c.fetchall()
    emb[None] = np.zeros(shape=embeddings[0].shape)
    for result in results:
        emb[result[1]] = embeddings[result[0]]
    embeddings = emb

def run_prediction_async(sender, data):
    global embeddings, characters
    model_path = get_value('model_path')
    db_path = get_value('db_path')
    if model_path == None or model_path == '':
        set_value('start_errors', 'No model specified')
        return
    if db_path == None or db_path == '':
        set_value('start_errors', 'No DB specified')
        return
    model = tf.keras.models.load_model(model_path, custom_objects={'SpecialEmbedding': SpecialEmbedding})
    db = sqlite3.connect(db_path)
    get_all_embeddings(db, model)
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
    log_info('Connected to chat. Ready for inference')
    bof_len = len('Betting open for: ')
    cur_pred = -1
    num_matches = 0
    num_correct = 0
    while True:
        msg = debug_line
        #msg = sock.recv(2048).decode('utf-8')
        if msg.startswith('PING'):
            sock.send('PONG\n'.encode('utf-8'))
        elif len(msg) > 0:
            if 'PRIVMSG' in msg:
                username, channel, message = re.search(':(.*)\!.*@.*\.tmi\.twitch\.tv PRIVMSG #(.*) :(.*)', msg).groups()
                if username == 'spriteclub':
                    if 'Betting open' in message:
                        log_debug(message)
                        if '(Exhibition by ' in message:
                            log_warning('Cannot do exhibition matches (yet...)')
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
                        set_team_values({'blue_raw': raw_blue, 'blue_0': split_blue[0], 'blue_1': split_blue[1], 'blue_2': split_blue[2], 'blue_3': split_blue[3]})
                        set_team_values({'red_raw': raw_red, 'red_0': split_red[0], 'red_1': split_red[1], 'red_2': split_red[2], 'red_3': split_red[3]})
                        if match_array is None:
                            log_warning('Something went wrong with team parsing')
                            continue
                        predictions = model(np.asarray([match_array]))[0].numpy()
                        cur_pred = np.argmax(predictions)
                        if cur_pred == 0:
                            tex_pred = ' blue'
                        elif cur_pred == 1:
                            tex_pred = ' red'
                        else:
                            tex_pred = ' at your own risk'
                        set_team_values({'blue_pred': predictions[0], 'red_pred': predictions[1], 'dis_pred': predictions[2]})
                        update_current_details()

                    elif 'Winner: ' in message:
                        if cur_pred == -1:
                            continue
                        if ('blue team wins!' in message and cur_pred == 0) or ('red team wins!' in message and cur_pred == 1):
                            set_value('total_correct', get_value('total_correct') + 1)
                        set_value('total_matches', get_value('total_matches') + 1)
                        set_value('total_accuracy', float(get_value('total_correct')/float(get_value('total_matches'))))
        return

def update_current_details():
    clear_table('CharTable')
    character_vals = [get_value('blue_0'), get_value('blue_1'), get_value('blue_2'), get_value('blue_3'), get_value('red_0'), get_value('red_1'), get_value('red_2'), get_value('red_3')]
    emb_to_plot = []
    for idx, character in enumerate(character_vals):
        if character != 'None':
            row_contents = [character]
            for x in embeddings[character]:
                row_contents.append(x)
            add_row('CharTable', row_contents)
            # Something like this: set_item_color(get_row('CharTable', -1), [0,0,255] if idx < 4 else [255,0,0]
    

def prediction_return_handler(sender, data):
    set_value('running', False)

def select_model(sender, data):
    open_file_dialog(callback=apply_model, extensions='.hdf5,.*')

def apply_model(sender, data):
    model_path = os.path.join(data[0], data[1])
    set_value('model_path', model_path)

def select_db(sender, data):
    open_file_dialog(callback=apply_db, extensions='.db,.*')

def apply_db(sender, data):
    db_path = os.path.join(data[0], data[1])
    set_value('db_path', db_path)

with window('Main Window'):
    with group('Load'):
        add_button('Select Model', callback=select_model)
        add_text('Model path: ')
        add_same_line()
        add_label_text('##modelpath', source='model_path', color=[0,255,0])
        add_button('Select DB', callback=select_db)
        add_text('DB path: ')
        add_same_line()
        add_label_text('##dbpath', source='db_path', color=[0,255,0])
        add_button('Start', callback=start_up)
        add_label_text('##failedstart', source='start_errors', color=[255,0,0])
        add_label_text('##connected', source='start_connect', color=[0,255,0])
        add_value('running', False)
    with group('Teams'):
        add_input_text('Blue Raw', readonly=True, source='blue_raw')
        add_same_line()
        add_input_text('Red Raw', readonly=True, source='red_raw')
        add_input_text('Blue 0', readonly=True, source='blue_0')
        add_same_line()
        add_input_text('Red 0', readonly=True, source='red_0')
        add_input_text('Blue 1', readonly=True, source='blue_1')
        add_same_line()
        add_input_text('Red 1', readonly=True, source='red_1')
        add_input_text('Blue 2', readonly=True, source='blue_2')
        add_same_line()
        add_input_text('Red 2', readonly=True, source='red_2')
        add_input_text('Blue 3', readonly=True, source='blue_3')
        add_same_line()
        add_input_text('Red 3', readonly=True, source='red_3')

    with group('Predictions'):
        add_input_text('Blue Probability', readonly=True, source='blue_pred')
        add_input_text('Red Probability', readonly=True, source='red_pred')
        add_input_text('Disaster Probability', readonly=True, source='dis_pred')
        add_separator()
        add_input_int('Total Matches', readonly=True, default_value=0, source='total_matches')
        add_input_int('Total Correct', readonly=True, default_value=0, source='total_correct')
        add_input_float('Total Accuracy', readonly=True, default_value=0.0, source='total_accuracy')

show_logger()

start_dearpygui(primary_window='Main Window')