import keras
import tensorflow.compat.v1 as tf
import numpy as np
from matplotlib import pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import CarliniL0Method
from art.attacks.evasion import CarliniLInfMethod
from art.attacks.evasion import DeepFool
from art.estimators.classification import TensorFlowClassifier
from art.estimators.classification import TensorFlowV2Classifier
from data.setup_mnist import MNIST
from models.carlini_models import carlini_mnist_model
from utils import my_utils as ut
from data.cifar10 import CIFAR10Dataset
from models.densenet_models import densenet_cifar10_model
from models.densenet_models import get_densenet_weights_path


ut.limit_gpu_usage()

#LOAD CIFAR_________________________________________________________________________________________________

CIFAR = CIFAR10Dataset()

(x_test, y_test) = CIFAR.get_test_dataset()
(x_train, y_train) = CIFAR.get_train_dataset()
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

#LOAD MODEL (DENSENET CIFAR)________________________________________________________________________________

model = densenet_cifar10_model(logits=True, input_range_type=1, pre_filter=lambda x:x)
model_weights_fpath = get_densenet_weights_path()
model.load_weights(model_weights_fpath)
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])

classifier = TensorFlowV2Classifier(
    model=model,
    input_shape=(32,32,3),
    loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.0,reduction="auto"),
    clip_values=(0, 1),
    nb_classes=10
)

y_pred = np.argmax(classifier.predict(x_test), axis=1)
y_bool = (y_pred == y_test[:len(y_pred)])

correct_idx = [i for i, val in enumerate(y_bool) if val]

attack = FastGradientMethod(estimator=classifier, eps=0.0156)

x_test_adv = attack.generate(x=x_test)

y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
y_bool_adv = (y_pred_adv == y_test[:len(y_pred_adv)])
res = [i for i, val in enumerate(y_bool_adv) if not val]
arr = np.asarray(res)
inter = np.intersect1d(arr, correct_idx)

atk_samples = x_test_adv[inter]

print(atk_samples.shape)

np.save("fgsm_cifar_index", inter)
np.save("fgsm_cifar_samples", atk_samples)



