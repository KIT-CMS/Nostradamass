from os import environ
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys, os
import numpy as np
seed = 1234
np.random.seed(seed)


from common_functions import add_pu_target
from common_functions import transform_fourvector
from common_functions import load_from_root, load_model, load_from_pickle

def train_model(X, Y,  channel, model_filename = "toy_mass.h5", out_folder='', previous_model=None,):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    from keras.layers import GaussianNoise
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    set_session(sess)
    kernel_initializer = "random_uniform"
    bias_initializer = "Zeros"
    #X, Y = add_pu_target(X, Y, 6., 24.)
    X, Y = add_pu_target(X, Y, 0., 0.)

    if channel == "tt":
        from losses import loss_fully_hadronic as loss
    elif channel == "mt" or channel == "et":
        from losses import loss_semi_leptonic as loss
    else:
        from losses import loss_fully_leptonic as loss
        
 
    if previous_model == None:    
        model = Sequential()
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(X.shape[1],)))
        model.add(GaussianNoise(stddev=1.0))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(Y.shape[1], activation='linear'))
        model.compile(loss=loss, optimizer='Adamax')
    else:
        model = load_model(previous_model)

    model.summary()
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(patience = 50)

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    tmp_X = X_train 

    for i in range(1):
        model_checkpoint = ModelCheckpoint( os.path.join(out_folder, 'model.'+str(i)+'.{epoch:04d}-{val_loss:.2f}.hdf5'),
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
        model.fit(tmp_X, Y_train,
                    batch_size=50000,
                    epochs=2000,
                    validation_data = (X_test, Y_test),
                    callbacks = [model_checkpoint, early_stopping])
    files = sorted([f for f in os.listdir(out_folder) if f.split(".")[-1] == "hdf5"])[0:-1]
    for f in files:
        os.remove(os.path.join(out_folder, f))
#    model.save(os.path.join(out_folder, model_filename))
    return model
    
if __name__ == '__main__':
    channel = sys.argv[1]
    out_folder = sys.argv[2]
    in_filenames = sys.argv[3:]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    f, ext = os.path.splitext(in_filenames[0])
    if len(in_filenames) and ext == ".pkl":
        X, Y, B, M, L, phys_M = load_from_pickle(in_filenames[0])
    else:
        X, Y, B, M, L, phys_M = load_from_root(in_filenames, channel, out_folder = out_folder)

    model = train_model(X, Y, out_folder=out_folder, channel = channel)
    model.summary()
