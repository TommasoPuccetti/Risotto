import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Comment to see Tensorflow and Keras logs and warnings
import keras
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt  
import numpy as np 
from data.setup_mnist import MNIST
from utils import my_utils as ut
from models.cleverhans_models import cleverhans_mnist_model 
import sklearn
from sklearn.metrics import *
from detectors.feature_squeezing import FeatureSqueezingDetector
from detectors.magnet_mnist import MagNetDetector
from matplotlib import pyplot as plt


#SET KERAS SESSION____________________________________________________________________________________________________________________________________________

tf.disable_v2_behavior() 
ut.limit_gpu_usage()
sess = ut.load_tf_session()
keras.backend.set_learning_phase(0)
x = tf.placeholder(tf.float64, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float64, shape=(None, 1))

#LOAD DATASET: MNIST__________________________________________________________________________________________________________________________________________

MNIST = MNIST()

x_test = MNIST.test_data
y_test = np.argmax(MNIST.test_labels, axis=1)
x_train = MNIST.train_data
y_train = np.argmax(MNIST.train_labels, axis=1)
x_val = MNIST.validation_data
y_val = MNIST.validation_labels

#LOAD CLASSIFIER MODEL (MNIST CLEVERHANS)_____________________________________________________________________________________________________________________

model = cleverhans_mnist_model(logits=True, input_range_type=1, pre_filter=lambda x:x)
model_weights_fpath = "./models/trained_models/MNIST_cleverhans_retrain.keras_weights.h5"
model.load_weights(model_weights_fpath)
model.trainable = False
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
model.summary()

#PREDICTION ON NORMAL TEST SET________________________________________________________________________________________________________________________________

y_pred = np.argmax(model.predict(x_test),axis=1)
acc = accuracy_score(y_pred, y_test)

layers = model.layers

#EXTRACT LAYERS: Remove comment to extract the model layers during prediction on normal test set
#output = ut.extract_layers(layers, x_test, "x_test_mnist", 500)

print('Test accuracy on normal examples %.4f' % (acc))

#LOAD ATTACKS_________________________________________________________________________________________________________________________________________________
#Insert a path to samples and indexes of the generated attacks

idx = np.load("./attacks/mnist_2_carlini_index.npy")
y_att = y_test[idx]
x_att = np.load("./attacks/mnist_2_carlini_samples.npy")

print("Attack Set shape: ", x_att.shape)

#PREDICTION ON ATTACKS________________________________________________________________________________________________________________________________________

y_pred_att = np.argmax(model.predict(x_att), axis=1)
acc = accuracy_score(y_pred_att, y_att)

#EXTRACT LAYERS: Remove comment to extract the model layers during prediction on the loaded attack set
#output = ut.extract_layers(layers, x_att, "carlini_l2_mnist2", 500)

print('Test accuracy on attacks %.4f' % (acc))

#MAGNET_______________________________________________________________________________________________________________________________________________________

detector = MagNetDetector(model, "MagNet")
detector.train()

y_test_pred, y_test_pred_score = detector.test(x_test)

acc = ut.calculate_accuracy_bool(y_test_pred)

print("MagNet FPR calculated on the MNIST test set", acc)

y_att_pred, y_att_pred_score = detector.test(x_att)

acc = ut.calculate_accuracy_bool(y_att_pred)

print("MagNet detection accuracy on the attack set", acc)

#SQUEEZER____________________________________________________________________________________________________________________________________________________

detector = FeatureSqueezingDetector(model, "FeatureSqueezing?squeezers=bit_depth_1,median_filter_2_2&distance_measure=l1", "MNIST2")
detector.train(x_train, y_train)

y_test_pred, y_test_pred_score = detector.test(x_test)

acc = ut.calculate_accuracy_bool(y_test_pred)

print("Squeezer FPR calculated on the MNIST test set", acc)

y_att_pred, y_att_pred_score = detector.test(x_att)

acc = ut.calculate_accuracy_bool(y_att_pred)

print("Squeezer detection accuracy on the attack set", acc)