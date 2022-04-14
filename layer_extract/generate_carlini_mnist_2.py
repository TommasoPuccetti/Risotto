import keras
import tensorflow.compat.v1 as tf
import numpy as np
from matplotlib import pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import DeepFool
from art.estimators.classification import TensorFlowClassifier
from art.estimators.classification import TensorFlowV2Classifier
from data.setup_mnist import MNIST
from models.cleverhans_models import cleverhans_mnist_model
import utils.my_utils as ut

ut.limit_gpu_usage()

#LOAD MNIST_________________________________________________________________________________________________

MNIST = MNIST()

x_train = MNIST.train_data
y_train = np.argmax(MNIST.train_labels, axis=1)
x_test = MNIST.test_data
y_test = np.argmax(MNIST.test_labels, axis=1)
x_valid = MNIST.validation_data
y_valid = np.argmax(MNIST.validation_labels, axis=1)

x_test = x_test[0:5]
y_test = y_test[0:5]

#LOAD MODEL________________________________________________________________________________________________

model = cleverhans_mnist_model(logits=True, input_range_type=1, pre_filter=lambda x:x)
model_weights_fpath = "./models/trained_models/MNIST_cleverhans_retrain.keras_weights.h5"
model.load_weights(model_weights_fpath)
model.trainable = False
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
model.summary()

classifier = TensorFlowV2Classifier(
    model=model,
    input_shape=(28,28,1),
    loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.0,reduction="auto"),
    clip_values=(0, 1),
    nb_classes=10
)

y_pred = np.argmax(classifier.predict(x_test), axis=1)
y_bool = (y_pred == y_test[:len(y_pred)])

correct_idx = [i for i, val in enumerate(y_bool) if val]

#GENERATE ATTACK___________________________________________________________________________________________

attack = CarliniL2Method(classifier=classifier, confidence=10, max_iter=1000, batch_size=500)

x_test_adv = attack.generate(x=x_test)

y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
y_bool_adv = (y_pred_adv == y_test[:len(y_pred_adv)])
res = [i for i, val in enumerate(y_bool_adv) if not val]
arr = np.asarray(res)
inter = np.intersect1d(arr, correct_idx)

atk_samples = x_test_adv[inter]

print(atk_samples.shape)

#SAVE ATTACK SET INDEX AND SAMPLES_________________________________________________________________________

np.save("carlini_mnist_2_index", inter)
np.save("carlini_mnist_2_samples", atk_samples)


