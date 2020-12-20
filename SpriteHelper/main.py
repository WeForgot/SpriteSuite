import os
import re
import socket
import sqlite3
import sys
import time

import numpy as np
from dotenv import load_dotenv
import tensorflow as tf

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

sys.path.append('..')
from NeuralClubbing.model import SpecialEmbedding
load_dotenv()

select_template = '''SELECT * FROM Characters WHERE Name=?'''

Winner_Hint = {
        0: 'Blue',
        1: 'Red',
        2: 'Crash',
        3: 'Skipped',
        4: 'Tie',
        5: 'Timeout'
    }

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
    
    red_turns = 0
    if ' / ' in red_str:
        red_arr = red_str.split(' / ')
    elif ' ⇒ ' in red_str:
        red_arr = red_str.split(' ⇒ ')
        red_turns = 1
    else:
        red_arr = [red_str]
    red_arr += [None] * (4 - len(red_arr))

    blue_went_bad = get_ids(blue_arr, conn)
    red_went_bad = get_ids(red_arr, conn)

    if blue_went_bad or red_went_bad:
        return None

    total_arr = blue_arr + [blue_turns] + red_arr + [red_turns]
    return total_arr

def main(model_name, db_name):
    model = tf.keras.models.load_model(model_name, custom_objects={'SpecialEmbedding': SpecialEmbedding})
    conn = sqlite3.connect(db_name)
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

    print('Setup done')

    bof_len = len('Betting open for: ')
    cur_pred = -1
    num_matches = 0
    num_correct = 0

    while True:
        msg = sock.recv(2048).decode('utf-8')
        if msg.startswith('PING'):
            sock.send('PONG\n'.encode('utf-8'))
        elif len(msg) > 0:
            if 'PRIVMSG' in msg:
                username, channel, message = re.search(':(.*)\!.*@.*\.tmi\.twitch\.tv PRIVMSG #(.*) :(.*)', msg).groups()
                if username == 'spriteclub':
                    if 'Betting open' in message:
                        to_strip_l = message.find('Betting open for: ') + len('Betting open for: ')
                        to_strip_r = message.rfind(' (')
                        message = message[to_strip_l:to_strip_r]
                        raw_blue, raw_red = message.split(' Vs. ')
                        if raw_blue.startswith('[ '):
                            raw_blue = raw_blue[2:-2]
                        if raw_red.startswith('[ '):
                            raw_red = raw_red[2:-2]
                        match_array = parse_teams(raw_blue, raw_red, conn)
                        if match_array is None:
                            print('Something went wrong with team parsing')
                            continue
                        predictions = model(np.asarray([match_array]))[0].numpy()
                        cur_pred = np.argmax(predictions)
                        print('------------------------------------------------')
                        print('{}: {}'.format(raw_blue, predictions[0]))
                        print('{}: {}'.format(raw_red, predictions[1]))
                        print('Everything is bad: {}'.format(predictions[2]))
                        if cur_pred < 2:
                            print('Bet {}'.format(Winner_Hint[cur_pred]))
                        else:
                            print('Possible dumpster fire. Ignore...')
                    elif 'Winner: ' in message:
                        if ('blue team wins!' in message and cur_pred == 0) or ('red team wins!' in message and cur_pred == 1):
                            num_correct += 1
                        num_matches += 1
                        print('Total correct: {}, Total matches: {}, Accuracy: {:.2f}'.format(num_correct, num_matches, float(num_correct) / float(num_matches)))
                        

if __name__ == '__main__':
    db_path = os.path.join('..','MatchScraper','sprite.db')
    model_path = os.path.join('..','NeuralClubbing','checkpoints','sprite_10_16_16_16_675.hdf5')
    main(model_path, db_path)