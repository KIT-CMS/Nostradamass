from os import environ
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#environ['CUDA_VISIBLE_DEVICES'] = "3"
import sys, os
import numpy as np
import json
seed = 1234
np.random.seed(seed)

from common_functions import add_pu_target
from common_functions import transform_fourvector
from common_functions import load_from_root, load_model, load_from_pickle


dataset_channel_dict = {
    "mt" : "SingleMuon",
    "et" : "SingleElectron",
    "tt" : "Tau",
    "em" : "MuonEG"
}

def train_model(X, Y,  channel="mt", metinputfile="metdata/metcovariance.root", model_filename = "toy_mass.h5", out_folder='', previous_model=None,):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, GaussianNoise
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    sess = tf.Session(config=config)
    set_session(sess)
    kernel_initializer = "random_uniform"
    bias_initializer = "Zeros"
    print "adding pu target"
    # Adjust the parameters to the expected MET-resolution scenario, derived from the MET covariance matrix of the data to be analyzed
    smearfile = "metdata/fullsmearing_22_11_2018_{CH}.npz".format(CH=channel)
    X, Y = add_pu_target(X, Y, dataset_channel_dict[channel], metinputfile, smearfile)
    #X, Y = add_pu_target(X, Y, 0., 0., 0.)

    if channel == "tt":
        from losses import loss_fully_hadronic as loss
    elif channel == "mt" or channel == "et":
        from losses import loss_semi_leptonic as loss
    else:
        from losses import loss_fully_leptonic as loss
    from losses import loss_dM_had, loss_dMtaus_had, loss_dPTtaus, loss_dPtaus, loss_dxyz, loss_dmet
    
    from keras.optimizers import Adamax 
    optimizer = Adamax()
    if previous_model == None:    
        model = Sequential()
        model.add(Dense(500, activation='linear', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(X.shape[1],)))
        model.add(GaussianNoise(stddev=1.0))
        for l in range(9):
            model.add(Dense(500, activation='elu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(Y.shape[1], activation='linear'))
        model.compile(loss=loss, optimizer=optimizer, metrics = [loss_dM_had, loss_dMtaus_had, loss_dPTtaus, loss_dPtaus, loss_dxyz, loss_dmet])
    else:
        model = load_model(previous_model)

    model.summary()
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping
    from keras.callbacks import TensorBoard
    #tensorboard = TensorBoard(log_dir=os.path.join(out_folder,'logs'), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    early_stopping = EarlyStopping(patience = 50)
    history_list = []

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42)
    tmp_X = X_train 

    for i in range(1):
        model_checkpoint = ModelCheckpoint( os.path.join(out_folder, 'model.'+str(i)+'.{epoch:04d}-{val_loss:.4f}.hdf5'),
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
        history_list.append(model.fit(tmp_X, Y_train,
                    batch_size=10000,
                    epochs=2000,
                    validation_data = (X_test, Y_test),
                    callbacks = [model_checkpoint, early_stopping]))
    with open(os.path.join(out_folder,"history.json"),"w") as hist:
        hist.write(json.dumps(history_list[-1].history, indent=2, sort_keys=True))
    files = sorted([f for f in os.listdir(out_folder) if f.split(".")[-1] == "hdf5"])[0:-1]
    # clean models that in the end turned out to be not the best
    for f in files:
        os.remove(os.path.join(out_folder, f))
    # saving not necessary, the best ones have been saved by the callback
    # model.save(os.path.join(out_folder, model_filename))
    return model
    
if __name__ == '__main__':
    channel = sys.argv[1]
    out_folder = sys.argv[2]
    in_filenames = sys.argv[3:]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    f, ext = os.path.splitext(in_filenames[0])
    X, Y, B, L = load_from_root(in_filenames, channel, use_jets=0)

    model = train_model(X, Y, out_folder=out_folder, channel = channel)
    model.summary()
