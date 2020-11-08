import os
import re
import socket
import sqlite3

import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
import tensorflow_addons as tfa

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

select_template = '''SELECT * FROM Characters WHERE Name=?'''

def one_hot_encode(val, max_val=5):
    hot = [0] * max_val
    # We assume values are NOT zero indexing anything
    hot[int(val)-1] = 1
    return hot

def main_driver(model_name, db_name, driver_loc):
    model = tf.keras.models.load_model(model_name)
    conn = sqlite3.connect(db_name)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('log-level=3')
    driver = webdriver.Chrome(driver_loc, options=options)
    driver.get('https://mugen.spriteclub.tv/')
    done = False
    while not done:
        _ = input('Press ENTER to read participants')
        blue_name = driver.find_element_by_id('bluePlayer').text
        red_name = driver.find_element_by_id('redPlayer').text
        red_info = conn.execute('''SELECT * FROM Characters WHERE Name= ?''', (red_name,)).fetchone()
        blue_info = conn.execute('''SELECT * FROM Characters WHERE Name= ?''', (blue_name,)).fetchone()
        if red_info is None or blue_info is None:
            print('Incorrect inputs')
            continue
        features = {}
        features['embeddings'] = np.asarray([[blue_info[0], red_info[0]]])
        features['divisions'] = np.asarray([[one_hot_encode(blue_info[2]), one_hot_encode(red_info[2])]])
        features['stats'] = np.asarray([[blue_info[3:], red_info[3:]]])
        pred = model.predict(features)[0]
        print('{}: {}\n{}: {}'.format(blue_name, pred[0], red_name, pred[1]))

def predict_one(model_name, db_name, red_name, blue_name):
    model = tf.keras.models.load_model(model_name)
    conn = sqlite3.connect(db_name)
    red_info = conn.execute('''SELECT * FROM Characters WHERE Name= ?''', (red_name,)).fetchone()
    blue_info = conn.execute('''SELECT * FROM Characters WHERE Name= ?''', (blue_name,)).fetchone()
    if red_info is None or blue_info is None:
        print('Incorrect inputs')
        return
    features = {}
    features['embeddings'] = np.asarray([[blue_info[0], red_info[0]]])
    features['divisions'] = np.asarray([[one_hot_encode(blue_info[2]), one_hot_encode(red_info[2])]])
    features['stats'] = np.asarray([[blue_info[3:], red_info[3:]]])
    pred = model.predict(features)[0]
    print('{}: {}\n{}: {}'.format(blue_name, pred[0], red_name, pred[1]))

#  Betting open for: Est Vs. Decade EX (~1469 rated matchmaking)
#   Length of beginning: 18
#   Length from end to parenthesis: 25
#   Could be 3 digits on rating but if we do only 25 and then strip trailing white space then we will get the correct size

def main(model_name, db_name):
    model = tf.keras.models.load_model(model_name)
    conn = sqlite3.connect(db_name)
    load_dotenv()
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

    red_name = None
    blue_name = None
    prediction = None
    correct = 0
    total = 0

    while True:
        msg = sock.recv(2048).decode('utf-8')
        if msg.startswith('PING'):
            sock.send('PONG\n'.encode('utf-8'))
        elif len(msg) > 0:
            if 'PRIVMSG' in msg:
                #  Betting open for: Est Vs. Decade EX (~1469 rated matchmaking)
                #   Length of beginning: 18
                #   Length from end to parenthesis: 25
                #   Could be 3 digits on rating but if we do only 25 and then strip trailing white space then we will get the correct size
                #  Betting open for: Lieselotte Achenbach Vs. Faust (3rd Division matchmaking)
                #  Betting open for: Nazo Vs. MVC2 Megaman (3rd Division tournament match)
                #  Betting open for: Brick Dragon II Vs. Schatten Geist (~1685 rated tournament match)
                #  GRAND FINALS! Betting open for: Gato XI Vs. Edward FA (5th Division tournament match)
                username, channel, message = re.search(':(.*)\!.*@.*\.tmi\.twitch\.tv PRIVMSG #(.*) :(.*)', msg).groups()
                if username == 'spriteclub':
                    if 'Betting open' in message:
                        if not '⇒' in message and not ' / ' in message:
                            if 'rated matchmaking)' in message:
                                names = message[18:-26].strip().split(' Vs. ')
                            elif 'GRAND FINALS!' in message and 'Division tournament match)' in message:
                                names = message[32:-31].strip().split(' Vs. ')
                            elif 'Division matchmaking)' in message:
                                names = message[18:-27].strip().split(' Vs. ')
                            elif ' Division tournament match)' in message:
                                names = message[18:-32].strip().split(' Vs. ')
                            elif ' rated tournament match)' in message:
                                names = message[18:-30].strip().split(' Vs. ')
                            else:
                                continue
                            red_info = conn.execute('''SELECT * FROM Characters WHERE Name= ?''', (names[1],)).fetchone()
                            red_name = names[1]
                            blue_info = conn.execute('''SELECT * FROM Characters WHERE Name= ?''', (names[0],)).fetchone()
                            blue_name = names[0]
                            if red_info is None or blue_info is None:
                                print('Incorrect inputs')
                                continue
                            features = {}
                            features['embeddings'] = np.asarray([[blue_info[0], red_info[0]]])
                            features['divisions'] = np.asarray([[one_hot_encode(blue_info[2]), one_hot_encode(red_info[2])]])
                            features['stats'] = np.asarray([[blue_info[3:], red_info[3:]]])
                            pred = model.predict(features)[0]
                            prediction = pred.tolist().index(max(pred.tolist()))
                            print('--------------------------------------------------------')
                            print('{}: {}\n{}: {}'.format(names[0], pred[0], names[1], pred[1]))
                            print('--------------------------------------------------------')
                        else:
                            print('Team fight, no can do')
                    elif 'Winner:' in message:
                        if red_name is None or blue_name is None or prediction is None:
                            continue
                        if (blue_name in message and prediction == 0) or (red_name in message and prediction == 1):
                            correct += 1
                        total += 1
                        print('Prediction rate: {}'.format(float(correct)/total))
                        red_name = None
                        blue_name = None
                        prediction = None




if __name__ == '__main__':
    #main('sprite.hdf5', 'sprite.db',os.path.join('..','utils','chromedriver'))
    db_path = os.path.join('..','MatchScraper','sprite.db')
    main('sprite.hdf5', db_path)