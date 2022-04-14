import numpy as np
import keras.backend as K
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from collections import Counter
import tensorflow.compat.v1 as tf 
import os 
import pickle


def load_tf_session():
    
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    
    # Create TF session and set as Keras backend session
    sess = tf.Session()
    
    print("Created TensorFlow session and set Keras backend.")

    return sess

def extract_layers(layers, X, who_data, batch):
    
    limit=X.shape[0]
    BATCH=batch
    x_batch = tf.data.Dataset.from_tensor_slices(X).batch(BATCH)

    for i in range(1, len(layers)):
        
        j=0
        
        if("activation" not in layers[i].name and "concatenate" not in layers[i].name and "batch_norm" not in layers[i].name ):
            
            while j <= limit-BATCH:             
                
                x_batch = X[j:j+BATCH]
                get_output = K.function([layers[0].input], [layers[i].output])
                conv_outputs = get_output(x_batch)
                j+=BATCH
                
                if(limit > j):
                    
                    X_batch = X[j-BATCH :  ]
                    get_output = K.function([layers[0].input], [layers[i].output])
                    conv_outputs = get_output(x_batch)
                
                p = Path("./layers/"+ str(i) + "_" +layers[i].name + "_batch_" + str(j) + "_" + str(who_data) + ".npy")
                f=p.open('ab')
                np.save(f, conv_outputs[0])
                f.close()

    return conv_outputs

def limit_gpu_usage():

    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def calculate_accuracy_bool(y_pred_bool):
    
    count = 0 
    
    for i in y_pred_bool:
        if( i == True):
            count = count + 1 
            
    acc = count/len(y_pred_bool)
    
    return acc

if __name__ == '__main__':
	print("ciao")
