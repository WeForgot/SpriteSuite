import os
import pickle
import sqlite3

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

def one_hot_encode(val, max_val):
    hot = [0] * max_val
    # We assume values are NOT zero indexing anything
    hot[int(val)-1] = 1
    return hot


def load_data(db_name):
    conn = sqlite3.connect(db_name)
    characters = {}
    for row in conn.execute('SELECT * FROM Characters'):
        cdx, name, division, life, power, attack, defense = row
        characters[cdx] = {'Name': name, 'Division': division, 'Life': life, 'Power': power, 'Attack': attack, 'Defense': defense}
    character_count = len(characters)
    features_idx = []
    features_div = []
    features_sta = []
    labels = []
    for row in conn.execute('SELECT * FROM Matches'):
        mdx, red, blue, winner, mode = row
        red_data = characters[red]
        blue_data = characters[blue]
        features_idx.append([blue, red])
        red_division = one_hot_encode(red_data['Division'], 5)
        blue_division = one_hot_encode(blue_data['Division'], 5)
        features_div.append([blue_division, red_division])
        red_stats = [red_data['Life'], red_data['Power'], red_data['Attack'], red_data['Defense']]
        blue_stats = [blue_data['Life'], blue_data['Power'], blue_data['Attack'], blue_data['Defense']]
        features_sta.append([blue_stats, red_stats])
        labels.append([1, 0] if blue == winner else [0, 1])
    return character_count + 1, {'embeddings': np.asarray(features_idx), 'divisions': np.asarray(features_div), 'stats': np.asarray(features_sta)}, np.asarray(labels)



def build_model(vocab_size, embedding_weights=None, embedding_size=10):
    # Two characters, one index for each
    input_emb = tf.keras.layers.Input(shape=(2,), name='embeddings')
    if embedding_weights is None:
        emb = tf.keras.layers.Embedding(vocab_size, embedding_size)(input_emb)
    else:
        emb = tf.keras.layers.Embedding(vocab_size, embedding_size, embeddings_initializer=tf.keras.initializers.Constant(embedding_weights))(input_emb)
    f_emb = tf.keras.layers.Flatten()(emb)
    #d_emb = tf.keras.layers.Dense(units=8, activation=tfa.activations.gelu)(f_emb)

    d_emb = tf.keras.layers.Dense(units=8, activation=None)(f_emb)
    b_emb = tf.keras.layers.BatchNormalization()(d_emb)
    a_emb = tf.keras.layers.PReLU()(b_emb)
    r_emb = tf.keras.layers.Dropout(0.4)(a_emb)
    

    #b_emb = tf.keras.layers.BatchNormalization()(d_emb)
    #r_emb = tf.keras.layers.Dropout(0.4)(b_emb)



    # Two characters, five possible divisions each one hot encoded
    input_div = tf.keras.layers.Input(shape=(2,5), name='divisions')
    f_div = tf.keras.layers.Flatten()(input_div)

    d_div = tf.keras.layers.Dense(units=8, activation=None)(f_div)
    b_div = tf.keras.layers.BatchNormalization()(d_div)
    a_div = tf.keras.layers.PReLU()(b_div)
    r_div = tf.keras.layers.Dropout(0.4)(a_div)

    #d_div = tf.keras.layers.Dense(units=8, activation=tfa.activations.gelu)(f_div)
    #b_div = tf.keras.layers.BatchNormalization()(d_div)
    #r_div = tf.keras.layers.Dropout(0.4)(b_div)

    # Two characters, 4 stats each
    input_sta = tf.keras.layers.Input(shape=(2,4), name='stats')
    f_sta = tf.keras.layers.Flatten()(input_sta)

    d_sta = tf.keras.layers.Dense(units=8, activation=None)(f_sta)
    b_sta = tf.keras.layers.BatchNormalization()(d_sta)
    a_sta = tf.keras.layers.PReLU()(b_sta)
    r_sta = tf.keras.layers.Dropout(0.4)(a_sta)

    #d_sta = tf.keras.layers.Dense(units=8, activation=tfa.activations.gelu)(f_sta)
    #b_sta = tf.keras.layers.BatchNormalization()(d_sta)
    #r_sta = tf.keras.layers.Dropout(0.4)(b_sta)

    # Combine inputs using addition
    #com = tf.keras.layers.Add()([r_emb, r_div, r_sta])

    # Combine inputs using concatenation
    com = tf.keras.layers.Concatenate()([r_emb, r_div, r_sta])
    d_com = tf.keras.layers.Dense(units=8, activation=None)(com)
    b_com = tf.keras.layers.BatchNormalization()(d_com)
    a_com = tf.keras.layers.PReLU()(b_com)
    r_com = tf.keras.layers.Dropout(0.4)(a_com)
    d_pre = tf.keras.layers.Dense(units=2, activation='softmax')(r_com)

    #d_com = tf.keras.layers.Dense(units=8, activation=tfa.activations.gelu)(com)
    #b_pre = tf.keras.layers.BatchNormalization()(d_com)
    #r_pre = tf.keras.layers.Dropout(0.4)(b_pre)
    #d_pre = tf.keras.layers.Dense(units=2, activation='softmax')(r_pre)

    model = tf.keras.Model(inputs=[input_emb,input_div,input_sta], outputs=[d_pre])
    model.summary()
    #lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1,  decay_steps=10000, decay_rate=0.96, staircase=True)
    lr = tfa.optimizers.Triangular2CyclicalLearningRate(1e-3, 1e-1, step_size=10000)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    #optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-1)
    loss = tf.keras.losses.CategoricalCrossentropy()
    #loss = tfa.losses.SparsemaxLoss()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model

def build_basic_model(vocab_size):
    # Two characters, one index for each
    input_emb = tf.keras.layers.Input(shape=(2,), name='embeddings')
    emb = tf.keras.layers.Embedding(vocab_size, 10)(input_emb)
    f_emb = tf.keras.layers.Flatten()(emb)
    d_emb = tf.keras.layers.Dense(units=8, activation=tfa.activations.gelu)(f_emb)
    b_emb = tf.keras.layers.BatchNormalization()(d_emb)
    r_emb = tf.keras.layers.Dropout(0.4)(b_emb)

    # Combine inputs using addition
    b_com = tf.keras.layers.BatchNormalization()(r_emb)
    d_com = tf.keras.layers.Dense(units=8, activation=tfa.activations.gelu)(b_com)
    b_pre = tf.keras.layers.BatchNormalization()(d_com)
    r_pre = tf.keras.layers.Dropout(0.4)(b_pre)
    d_pre = tf.keras.layers.Dense(units=2, activation=tfa.activations.sparsemax)(r_pre)

    model = tf.keras.Model(inputs=input_emb, outputs=d_pre)
    model.summary()
    #lr = tfa.optimizers.Triangular2CyclicalLearningRate(initial_learning_rate=1e-3, maximal_learning_rate=1e0, step_size=1e3)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.5, nesterov=True)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-1)
    #loss = tf.keras.losses.CategoricalCrossentropy()
    loss = tfa.losses.SparsemaxLoss() # This apparently does REALLY FUCKING GOOD
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model

def build_joined_model(vocab_size, embedding_weights=None):
    # Two characters, one index for each
    input_emb = tf.keras.layers.Input(shape=(2,), name='embeddings')
    if embedding_weights is None:
        emb = tf.keras.layers.Embedding(vocab_size, 10)(input_emb)
    else:
        emb = tf.keras.layers.Embedding(vocab_size, 10, embeddings_initializer=tf.keras.initializers.Constant(embedding_weights), trainable=False)(input_emb)
    blue_emb = emb[:,0,:]
    red_emb = emb[:,1,:]
    d_emb = tf.keras.layers.Dense(units=4, activation=None)
    bd_emb = d_emb(blue_emb)
    rd_emb = d_emb(red_emb)
    de_con = tf.keras.layers.Concatenate()([bd_emb, rd_emb])
    b_emb = tf.keras.layers.BatchNormalization()(de_con)
    r_emb = tf.keras.layers.Dropout(0.4)(b_emb)



    # Two characters, five possible divisions each one hot encoded
    input_div = tf.keras.layers.Input(shape=(2,5), name='divisions')
    blue_div = input_div[:,0,:]
    red_div = input_div[:,1,:]
    d_div = tf.keras.layers.Dense(units=4, activation=None)
    bd_div = d_div(blue_emb)
    rd_div = d_div(red_emb)
    dd_con = tf.keras.layers.Concatenate()([bd_div, rd_div])
    b_div = tf.keras.layers.BatchNormalization()(dd_con)
    r_div = tf.keras.layers.Dropout(0.4)(r_emb)


    # Two characters, 4 stats each
    input_sta = tf.keras.layers.Input(shape=(2,4), name='stats')
    blue_sta = input_sta[:,0,:]
    red_sta = input_sta[:,1,:]
    d_sta = tf.keras.layers.Dense(units=4, activation=None)
    bd_sta = d_sta(blue_sta)
    rd_sta = d_sta(red_sta)
    ds_con = tf.keras.layers.Concatenate()([bd_sta, rd_sta])
    b_sta = tf.keras.layers.BatchNormalization()(ds_con)
    r_sta = tf.keras.layers.Dropout(0.4)(b_sta)

    # Combine inputs using addition
    com = tf.keras.layers.Add()([r_emb, r_div, r_sta])
    d_com = tf.keras.layers.Dense(units=8, activation=tfa.activations.gelu)(com)
    b_pre = tf.keras.layers.BatchNormalization()(d_com)
    r_pre = tf.keras.layers.Dropout(0.4)(b_pre)
    d_pre = tf.keras.layers.Dense(units=2, activation='softmax')(r_pre)

    model = tf.keras.Model(inputs=[input_emb,input_div,input_sta], outputs=[d_pre])
    model.summary()
    #lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=10000, decay_rate=0.99, staircase=True)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-1)
    loss = tf.keras.losses.CategoricalCrossentropy()
    #loss = tfa.losses.SparsemaxLoss()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model

if __name__ == '__main__':
    load_old = False
    model_name = ''

    db_path = os.path.join('..','MatchScraper','sprite.db')
    tokens, features, labels = load_data(db_path)
    print('Total feature sets: {}'.format(len(labels)))
    if os.path.exists('output.log'):
        os.remove('output.log')
    for idx in [6]:
        if load_old:
            t = tf.keras.models.load_model(model_name)
        else:
            t = build_model(tokens, embedding_size=idx)
        callbacks = []
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=str(idx)+'-sprite-{val_loss:.4f}-{val_accuracy:.4f}.hdf5', monitor='val_loss', save_best_only=True))
        #t.fit(features, labels, epochs=5000, batch_size=64, validation_split=0.2, callbacks=callbacks) # Adamax
        t.fit(features, labels, epochs=5000, batch_size=256, validation_split=0.2, callbacks=callbacks) # SGD