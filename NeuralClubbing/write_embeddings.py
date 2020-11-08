import os
import sqlite3

import tensorflow as tf
import tensorflow_addons as tfa

if __name__ == '__main__':
    model = tf.keras.models.load_model(os.path.join('..','SpriteHelper','sprite.hdf5'))
    model.summary()
    weights = model.get_weights()[0]
    conn = sqlite3.connect(os.path.join('..','MatchScraper','sprite.db'))
    characters = conn.execute('SELECT * FROM Characters').fetchall()
    character_file = open(os.path.join('..','SpriteHelper','characters.tsv'), 'w', encoding='utf-8')
    character_file.write('Name\tDivision\n')
    weights_file = open(os.path.join('..','SpriteHelper','weights.tsv'), 'w', encoding='utf-8')
    for idx, row in enumerate(characters):
        vec = weights[idx]
        print(row[1])
        print(row[2])
        print('---------------------')
        character_file.write('{}\t{}\n'.format(row[1], row[2]))
        weights_file.write('\t'.join([str(x) for x in vec]) + '\n')
    weights_file.close()
    character_file.close()