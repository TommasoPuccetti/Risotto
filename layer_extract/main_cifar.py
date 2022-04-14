import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Comment to see Tensorflow and Keras logs and warnings
import keras
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow.compat.v1 as tf
import numpy as np 
from data.setup_cifar import CIFAR
from utils import my_utils as ut
import sklearn
from sklearn.metrics import *
from detectors.feature_squeezing import FeatureSqueezingDetector
from detectors.magnet_cifar import MagNetDetector as MagNetDetectorCIFAR
from matplotlib import pyplot as plt
from models.densenet_models import densenet_cifar10_model
from models.densenet_models import get_densenet_weights_path


#SET KERAS SESSION____________________________________________________________________________________________________________________________________________

tf.disable_v2_behavior() 
ut.limit_gpu_usage()
sess = ut.load_tf_session()
keras.backend.set_learning_phase(0)
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.float32, shape=(None, 1))

#LOAD DATASET: CIFAR-10_______________________________________________________________________________________________________________________________________

CIFAR = CIFAR()

x_test = CIFAR.test_data
y_test = np.argmax(CIFAR.test_labels, axis=1)
x_train = CIFAR.train_data
y_train = np.argmax(CIFAR.train_labels, axis=1)

#LOAD CLASSIFIER MODEL (CIFAR DENSENET)______________________________________________________________________________________________________________________

model = densenet_cifar10_model(logits=True, input_range_type=1, pre_filter=lambda x:x)
model_weights_fpath = get_densenet_weights_path()
model.load_weights(model_weights_fpath)
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
model.summary()

#PREDICTION ON NORMAL TEST SET_______________________________________________________________________________________________________________________________

y_pred = np.argmax(model.predict(x_test),axis=1)
acc = accuracy_score(y_pred, y_test)

layers = model.layers

#EXTRACT LAYERS: Remove comment to extract the model layers during prediction on normal test set
#output = ut.extract_layers(layers, x_test, "x_test_cifar", 500)

print('Test accuracy on normal examples %.4f' % (acc))

#LOAD ATTACKS_______________________________________________________________________________________________________________________________________________
#Insert a path to samples and indexes of the generated attacks

idx = np.load("./attacks/cifar_carlini_index.npy")
y_att = y_test[idx]
x_att = np.load("./attacks/cifar_carlini_samples.npy")

print("Attack Set shape: ", x_att.shape)

#PREDICTION ON ATTACKS______________________________________________________________________________________________________________________________________

y_pred_att = np.argmax(model.predict(x_att), axis=1)
acc = accuracy_score(y_pred_att, y_att)

#EXTRACT LAYERS: Remove comment to extract the model layers during prediction on the loaded attack set
#output = ut.extract_layers(layers, x_att, "x_att_cifar", 500)

print("Model accuracy on the loaded attack: ", acc)

#MAGNET______________________________________________________________________________________________________________________________________________________

detector = MagNetDetectorCIFAR(model, "MagNet")
detector.train(x_train, y_train)

y_test_pred, y_pred_score = detector.test(x_test)

acc = ut.calculate_accuracy_bool(y_test_pred)

print("MagNet FPR calculated on the CIFAR-10 test set", acc)

y_att_pred, y_att_pred_score = detector.test(x_att)

acc = ut.calculate_accuracy_bool(y_att_pred)

print("MagNet detection accuracy on the attack set", acc)

#SQUEEZER____________________________________________________________________________________________________________________________________________________

detector = FeatureSqueezingDetector(model, "FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1", "CIFAR")
detector.train(x_train, y_train)

y_test_pred, y_test_pred_score = detector.test(x_test)
acc = ut.calculate_accuracy_bool(y_test_pred)

print("Squeezer FPR calculated on the CIFAR-10 test set", acc)

y_att_pred, y_att_pred_score = detector.test(x_att)
acc = ut.calculate_accuracy_bool(y_att_pred)

print("Squeezer detection accuracy on the attack set", acc)








