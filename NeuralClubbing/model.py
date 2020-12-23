import os
from os.path import join
import sqlite3

import numpy as np
import tensorflow as tf

def one_hot_encode(val, max_val):
    hot = [0] * max_val
    hot[int(val)] = 1
    return hot

def load_data(db_name, collapse_degenerate_labels=True):
    conn = sqlite3.connect(db_name)
    features = []
    labels = []
    num_characters = conn.execute('SELECT COUNT(*) FROM Characters').fetchone()[0]
    for row in conn.execute('SELECT Blue_0, Blue_1, Blue_2, Blue_3, Blue_Turns, Red_0, Red_1, Red_2, Red_3, Red_Turns, Outcome FROM Matches'):
        if collapse_degenerate_labels:
            label = one_hot_encode(min(row[-1], 2),3)
        else:
            label = one_hot_encode(row[-1], 6)
        feature = [x if x is not None else 0 for x in row[:-1]]
        features.append(feature)
        labels.append(label)
    return num_characters+1, np.asarray(features), np.asarray(labels)

class SpecialEmbedding(tf.keras.layers.Layer):
    def __init__(self, emb_size, emb_len, latent_dim=16, **kwargs):
        super(SpecialEmbedding, self).__init__(**kwargs)
        self.emb_size = emb_size
        self.emb_len = emb_len
        self.latent_dim = latent_dim

        self.embedding = tf.keras.layers.Embedding(emb_size, emb_len, mask_zero=True)
        
        self.turns_rnn = tf.keras.layers.LSTM(latent_dim, activation='relu')
        self.teams_dns = tf.keras.layers.Dense(units=latent_dim, activation='relu')
        self.teams_gap = tf.keras.layers.GlobalMaxPooling1D()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.flat = tf.keras.layers.Flatten()
    
    def get_config(self):
        config = super(SpecialEmbedding, self).get_config()
        config.update({'emb_size': self.emb_size,
                'emb_len': self.emb_len,
                'latent_dim': self.latent_dim})
        return config

    
    def call(self, x):
        blue, bt, red, rt = tf.split(x, [4,1,4,1], axis=1)
        blue_emb = self.embedding(blue)
        blue_msk = self.embedding.compute_mask(blue)
        red_emb = self.embedding(red)
        red_msk = self.embedding.compute_mask(red)
        blue_team = self.teams_gap(self.teams_dns(blue_emb))
        blue_turn = self.turns_rnn(blue_emb, mask=blue_msk)
        blue_result = tf.where(bt == 0, blue_team, blue_turn)
        red_team = self.teams_gap(self.teams_dns(red_emb))
        red_turn = self.turns_rnn(red_emb, mask=red_msk)
        red_result = tf.where(rt == 0, red_team, red_turn)
        x = self.concat([blue_result, red_result])
        x = self.flat(x)
        return x

def build_model(emb_size, emb_len, output_dim, latent_dims):

    return tf.keras.Sequential([
        SpecialEmbedding(emb_size, emb_len, latent_dim=latent_dims[0], name='special_embedding'),
        tf.keras.layers.Dense(units=latent_dims[1], activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(units=latent_dims[2], activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(units=output_dim, activation='softmax')
    ])


def main():
    num_characters, features, labels = load_data(os.path.join('..','MatchScraper','sprite.db'), collapse_degenerate_labels=True)
    emb_len = [7,8,9,10,11,12]
    output_labels = 3
    for emb in emb_len:
        for latent_pack in [([16,32,16], '131'), ([16,32,32], '133'), ([16,16,16], '111'), ([16,16,32], '113')]:
            latent_dim, suffix = latent_pack
            model = build_model(num_characters, emb, output_labels, latent_dim)
            lr = tf.keras.experimental.CosineDecayRestarts(1e-2, 3000)
            optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
            loss = tf.keras.losses.CategoricalCrossentropy()
            model.compile(optimizer, loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])
            checkpoint_name = os.path.join('.','checkpoints', 'vl_{val_loss:.3f}_ca_{val_categorical_accuracy:.3f}' + '_{}_{}.hdf5'.format(emb, suffix))
            callbacks = [tf.keras.callbacks.TerminateOnNaN(), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_loss', save_best_only=True)]
            history = model.fit(x=features, y=labels, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, callbacks=callbacks)

if __name__ == '__main__':
    main()