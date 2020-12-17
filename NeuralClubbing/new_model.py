import os
import sqlite3

import numpy as np
import tensorflow as tf

def one_hot_encode(val, max_val):
    hot = [0] * max_val
    hot[int(val)] = 1
    return hot

def load_data(db_name):
    conn = sqlite3.connect(db_name)
    features = []
    labels = []
    num_characters = conn.execute('SELECT COUNT(*) FROM Characters').fetchone()[0]
    for row in conn.execute('SELECT Blue_0, Blue_1, Blue_2, Blue_3, Blue_Turns, Red_0, Red_1, Red_2, Red_3, Red_Turns, Outcome FROM Matches'):
        label = one_hot_encode(row[-1], 6)
        feature = [x if x is not None else 0 for x in row[:-1]]
        features.append(feature)
        labels.append(label)
    return num_characters+1, np.asarray(features), np.asarray(labels)



class NewModel(tf.keras.Model):
    def __init__(self, emb_size, emb_len=10, latent_dim=16):
        super(NewModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(emb_size, emb_len, mask_zero=True)
        self.turns_rnn = tf.keras.layers.LSTM(latent_dim)
        self.teams_dns = tf.keras.layers.Dense(units=latent_dim, activation='tanh')
        self.teams_gap = tf.keras.layers.GlobalAveragePooling1D()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.flat = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(units=latent_dim, activation=None)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.pre_1 = tf.keras.layers.PReLU()
        self.dp_1 = tf.keras.layers.Dropout(rate=0.4)

        self.dense_2 = tf.keras.layers.Dense(units=latent_dim, activation=None)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.pre_2 = tf.keras.layers.PReLU()
        self.dp_2 = tf.keras.layers.Dropout(rate=0.4)

        self.out = tf.keras.layers.Dense(units=6, activation='softmax')
    
    def call(self, x):
        blue, bt, red, rt = tf.split(x, [4,1,4,1], axis=1)
        blue_emb = self.embedding(blue)
        blue_msk = self.embedding.compute_mask(blue)
        red_emb = self.embedding(red)
        red_msk = self.embedding.compute_mask(red)
        blue_team = self.teams_gap(self.teams_dns(blue_emb), mask=blue_msk)
        blue_turn = self.turns_rnn(blue_emb, mask=blue_msk)
        blue_result = tf.where(bt == 0, blue_team, blue_turn)
        red_team = self.teams_gap(self.teams_dns(red_emb), mask=red_msk)
        red_turn = self.turns_rnn(red_emb, mask=red_msk)
        red_result = tf.where(bt == 0, red_team, red_turn)
        x = self.concat([blue_result, red_result])
        x = self.flat(x)

        x = self.dense_1(x)
        x = self.bn_1(x)
        x = self.pre_1(x)
        x = self.dp_1(x)

        x = self.dense_2(x)
        x = self.bn_2(x)
        x = self.pre_2(x)
        x = self.dp_2(x)

        x = self.out(x)
        return x


def main():
    num_characters, features, labels = load_data(os.path.join('..','MatchScraper','sprite.db'))
    emb_len = [8, 10, 12]
    for idx in range(2,3):
        for emb in emb_len:
            model = NewModel(num_characters, emb_len=emb)
            #lr = tf.keras.experimental.CosineDecayRestarts(1e-2, 3000)
            #optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
            optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-2)
            loss = tf.keras.losses.CategoricalCrossentropy()
            model.compile(optimizer, loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])
            callbacks = [tf.keras.callbacks.TerminateOnNaN(), tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=10, mode='max', restore_best_weights=True), tf.keras.callbacks.TensorBoard()]
            history = model.fit(x=features, y=labels, epochs=40, batch_size=32, validation_split=0.2, callbacks=callbacks)
            checkpoint_name = os.path.join('.','checkpoints', 'sprite_{}_{}_{}'.format(emb, idx, int(1000 * max(history.history['val_categorical_accuracy']))))
            model.save(checkpoint_name, save_format='tf')

if __name__ == '__main__':
    main()