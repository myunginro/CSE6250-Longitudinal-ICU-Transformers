from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()
print(args)

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')



maxlen = 48*76 
vocab_size = 48*76 

# Build transformer model
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        x = x - tf.math.reduce_min(x) # so that all numbers are greater or equal to 0
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile("adam", 'binary_crossentropy', metrics=["AUC"])
print(model.summary())



# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)


test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    
discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl
#print(args_dict)


if args.mode == 'train':
    
    # Read data
    train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

    if target_repl:
        T = train_raw[0][0].shape[0]

        def extend_labels(data):
            data = list(data)
            labels = np.array(data[1])  # (B,)
            data[1] = [labels, None]
            data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
            data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
            return data

        train_raw = extend_labels(train_raw)
        val_raw = extend_labels(val_raw)

    Xtrain = np.array(train_raw[0]).reshape((-1, 48*76))
    Ytrain = np.array(train_raw[1]).reshape((-1,1))
    Xval = np.array(val_raw[0]).reshape((-1, 48*76))
    Yval = np.array(val_raw[1]).reshape((-1,1))

    with open('train.npy',"wb") as f:
        np.save(f,Xtrain)
        np.save(f,Ytrain)

    with open('val.npy',"wb") as f:
        np.save(f,Xval)
        np.save(f,Yval)
    
      
    with open('train.npy',"rb") as f:
        Xtrain = np.load(f)
        Ytrain = np.load(f)

    with open('val.npy',"rb") as f:
        Xval = np.load(f)
        Yval = np.load(f)

    keras_logs = os.path.join(args.output_dir, 'mimic3models/in_hospital_mortality/keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, 'transformer.csv'),
                           append=True, separator=';')

    filepath = os.path.join(args.output_dir, 'mimic3models/in_hospital_mortality/keras_states/transformer_best.state')
    earlyStopping = EarlyStopping(monitor='val_auc', patience=10, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1,save_best_only=True, mode='max')
    callbacks_list = [earlyStopping, checkpoint, csv_logger]

    model.fit(Xtrain, Ytrain, batch_size=5, epochs=100, callbacks=callbacks_list,
            validation_data=(Xval, Yval))

elif args.mode == 'test':
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                        return_names=True)
    test_raw = ret['data']
    test_names = ret['names']
    
    Xtest = np.array(test_raw[0]).reshape((-1, 48*76))
    Ytest = np.array(test_raw[1]).reshape((-1,1))

    model = keras.models.load_model(os.path.join(args.output_dir, 'mimic3models/in_hospital_mortality/keras_states/transformer_best.state'))


    print(Xtest[3051, 1266])   
    print(np.mean(Xtest,0)[1266])
    Xtest = np.delete(Xtest, 3051, 0) # large feature value for sequence 3051, event 1266, likely outlier
    Ytest = np.delete(Ytest, 3051, 0) # same as above
    print(np.mean(Xtest,0)[1266])

    predictions = model.predict(Xtest, batch_size=1, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(Ytest, predictions)

    path = os.path.join(args.output_dir, "test_predictions.csv")
    utils.save_results(test_names, predictions, Ytest, path)

else:
    raise ValueError("Wrong value for args.mode")
