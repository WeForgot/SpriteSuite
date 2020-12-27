import os
from os.path import join
import sqlite3

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.eager.context import collect_graphs

def one_hot_encode(val, max_val):
    hot = [0] * max_val
    hot[int(val)] = 1
    return hot

# skip_bad_labels takes precidence over collapse_degenerate_labels. Should eventually turn this into an enum
def load_data(db_name, collapse_degenerate_labels=True, skip_bad_labels=False, make_binary=False, skip_exhibs=False):
    conn = sqlite3.connect(db_name)
    features = []
    labels = []
    # Grab all characters from the DB
    num_characters = conn.execute('SELECT COUNT(*) FROM Characters').fetchone()[0]
    # Inline grab of all matches in the DB
    for row in conn.execute('SELECT Blue_0, Blue_1, Blue_2, Blue_3, Blue_Turns, Red_0, Red_1, Red_2, Red_3, Red_Turns, Outcome, Session FROM Matches'):
        # Decide how we format the label
        if skip_exhibs and row[-1] == 'Exhibitions':
            continue
        if make_binary:
            if row[-2] > 1:
                continue
            label = row[-2]
        elif skip_bad_labels:
            if row[-2] > 1:
                continue
            label = one_hot_encode(row[-2],2)
        elif collapse_degenerate_labels:
            label = one_hot_encode(min(row[-2], 2),3)
        else:
            label = one_hot_encode(row[-2], 6)
        # Everything except the last row is are indicies for characters or 0's for padding
        feature = [x if x is not None else 0 for x in row[:-2]]
        features.append(feature)
        labels.append(label)
    # Return the number of characters as well for embedding layer reasons
    return num_characters+1, np.asarray(features), np.asarray(labels)


class SpecialEmbedding(tf.keras.layers.Layer):
    def __init__(self, emb_size, emb_len, latent_dim=16, **kwargs):
        super(SpecialEmbedding, self).__init__(**kwargs)
        self.emb_size = emb_size
        self.emb_len = emb_len
        self.latent_dim = latent_dim

        self.embedding = tf.keras.layers.Embedding(emb_size, emb_len, mask_zero=True)
        
        self.turns_rnn = tf.keras.layers.LSTM(latent_dim, activation='relu')
        #self.turns_rnn = tf.keras.layers.SimpleRNN(latent_dim, activation='relu')
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
        blue, bt, red, rt = tf.split(x, [4,1,4,1], axis=1) # Split vectors into 2 teams of four characters with one bit representing turns
        # Embed and mask characters
        blue_emb = self.embedding(blue)
        blue_msk = self.embedding.compute_mask(blue)
        red_emb = self.embedding(red)
        red_msk = self.embedding.compute_mask(red)

        # We still need to compute both types of matches (even if it isn't used)
        # We then replace the latent embedding depending on whether it is turns or not
        blue_team = self.teams_gap(self.teams_dns(blue_emb))
        blue_turn = self.turns_rnn(blue_emb, mask=blue_msk)
        blue_result = tf.where(bt == 0, blue_team, blue_turn)
        red_team = self.teams_gap(self.teams_dns(red_emb))
        red_turn = self.turns_rnn(red_emb, mask=red_msk)
        red_result = tf.where(rt == 0, red_team, red_turn)

        # Combine and flatten for processing by next layers
        x = self.concat([blue_result, red_result])
        x = self.flat(x)

        return x

def build_model(emb_size, emb_len, output_dim, latent_dims):

    # Remember, more simple can be better (in this case it is)
    return tf.keras.Sequential([
        SpecialEmbedding(emb_size, emb_len, latent_dim=latent_dims[0], name='special_embedding'),
        tf.keras.layers.Dense(units=latent_dims[1], activation=None),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(units=output_dim, activation='sigmoid' if output_dim == 1 else 'softmax')
    ])

def main():
    collapse_degenerate_labels = False # Reduce label count to 3 (softmax + categorical cross entropy)
    skip_bad_labels = True # Reduce label count to 2 (softmax + categorical cross entropy)
    # Primary improvement for binary is that we have less parameters on the final layer, the loss is comparable between 2 class and this
    make_binary = True # Turn problem into binary labels where 1 indicates red will win (sigmoid + binary cross entropy)
    skip_exhibs = True # Exhibs sometimes have pallete shenan's. Ignoring them is the more stable choice for embeddings overall
    label_smoothing = 0.0 # Leniency for "correctness" of a guess. Higher numbers = answers closer to 0.5
    optimizer_used = 'adamax' # Can be 'adamax' or 'novo'. Anything else defaults to SGD
    early_stop = True # Whether to add the early stopping callback (patiece can be set in code below)
    auto_checkpoint = True # Whether to let the model automatically create checkpoints every time it gets a new personal best
    batch_size = 32

    num_characters, features, labels = load_data(os.path.join('..','MatchScraper','sprite.db'), collapse_degenerate_labels=collapse_degenerate_labels, skip_bad_labels=skip_bad_labels, make_binary=make_binary, skip_exhibs=skip_exhibs)
    train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size=0.2, random_state=420)
    emb_len = [9]
    if make_binary:
        output_labels = 1
    elif skip_bad_labels:
        output_labels = 2
    elif collapse_degenerate_labels:
        output_labels = 3
    else:
        output_labels = 6
    for emb in emb_len:
        for latent_pack in [([64,64], '66')]:
            latent_dim, suffix = latent_pack
            model = build_model(num_characters, emb, output_labels, latent_dim)
            if optimizer_used == 'adamax':
                # Adamax consistently has great results
                optimizer = tf.keras.optimizers.Adamax(1e-3)
            elif optimizer_used == 'novo':
                # Novograd seems cool in theory but for some reason doesn't perform well on this, keeping it here for shits and giggles
                optimizer = tfa.optimizers.NovoGrad(1e-2, weight_decay=0.0, grad_averaging=False, amsgrad=False)
            else:
                # SGD + momentum + scheduler actually has yielded the best results, just needs multiple attempts
                lr = tf.keras.experimental.CosineDecayRestarts(1e-1, first_decay_steps=160000, alpha=0.01, t_mul=1, m_mul=1)
                optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)

            if make_binary:
                loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
                metrics = [tf.keras.metrics.BinaryAccuracy()]
                checkpoint_name = os.path.join('.','checkpoints', 'binary_vl_{val_loss:.3f}_ca_{val_binary_accuracy:.3f}' + '_{}_{}.hdf5'.format(emb, suffix))
            else:
                loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
                metrics = [tf.keras.metrics.CategoricalAccuracy()]
                checkpoint_name = os.path.join('.','checkpoints', 'vl_{val_loss:.3f}_ca_{val_categorical_accuracy:.3f}' + '_{}_{}.hdf5'.format(emb, suffix))
            model.compile(optimizer, loss, metrics=metrics)
            callbacks = [tf.keras.callbacks.TerminateOnNaN()] 
            if early_stop:
                callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))
            if auto_checkpoint:
                callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_loss', save_best_only=True))
            history = model.fit(x=train_features, y=train_labels, epochs=200, batch_size=batch_size, shuffle=True, validation_data=(valid_features,valid_labels), callbacks=callbacks)

if __name__ == '__main__':
    main()