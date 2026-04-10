import copy
import time
import datetime
import platform
import matplotlib.animation as animation
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv2D, UpSampling2D, Flatten, Dense, Reshape,
        Dropout, MaxPooling2D, Resizing
    )
except ImportError:
    print("Warning - COULD NOT IMPORT TENSORFLOW")

import matplotlib
os_name = platform.system()
if 'Linux' in os_name:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

# ------------------------------------------------------------------
# Static config
# ------------------------------------------------------------------
#datadir = '/Users/manavkalra/Downloads/PL-NN-testdata_forDec2025/'
datadir = '/home/manav//PL-NN-testdata_forDec2025/'
slmdatadir = datadir
outdir = datadir

save_filename_pref = 'pl2wf2psf_data202407_model01_'

load_precombined_PLims_filename = 'pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined'
precombined_psf_filename = None
load_precombined_wfims_filename = 'slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined'

addnoise_PL = None
addnoise_PSF = None
addnoise_WF = None

plim_cropcnt = [80, 103]
plim_cropsz = 128
mask_pupil = True
rescale_wfim = 0.188
convert_slm2rad = True
slmrange = 4*np.pi
psfim_cropcnt = [139, 134]
psfim_cropsz = 48
rescale_psfim = 2
remove_clock = True

keep_orig_psfs = False
stat_frms = 1000
testdatasplit = 0.2
shuffle_before_split = False
use_subset = None
num_preds = 100
do_subset_on_read = False
    
BASE_PDICT = {
    'actFunc': 'relu',
    'batchSize': 16,
    "learningRate": 6.814073755185045e-05,
    "lossFunc_psf": "mean_squared_error",
    "lossFunc_wf": "mean_squared_error",
    "epochs": 100,
    "dropout_rate": 0.05378660067364384,
    "dropout_rate_dense": 0.0019573719804583177,
    "dropout_rate_psf": 0.2726712051110854,
    'ksz_enc': 5,
    'ksz_psf': 3,
    'ksz_wf': 5,
    'nfilts_enc': 96,
    'nfilts_psf': 64,
    'nfilts_wf': 96,
    'loss_weight': 0.549,
    'n_units_dense': 4096,
    'enable_lr_sched': False,
    'reduceLR_start': 1e-4,
    'reduceLR_factor': 0.5,
    'reduceLR_patience': 5,
    'reduceLR_min_lr': 1e-6,
    'reduceLR_cooldown': 5,
}


_DATA_CACHE = None


def get_base_pdict():
    return copy.deepcopy(BASE_PDICT)


def plot_truepredims(true_im, pred_im, true_im2, pred_im2, camera_im=None, pausetime=0.01):
    if camera_im is None:
        ncol = 2
        offs = 0
    else:
        ncol = 3
        offs = 1

    plt.clf()
    plt.subplot(2, ncol, 1)
    plt.imshow(true_im)
    plt.title('True pupil phase')
    plt.colorbar()

    plt.subplot(2, ncol, 2)
    plt.imshow(pred_im)
    plt.title('Predicted pupil phase')
    plt.colorbar()

    plt.subplot(2, ncol, 3 + offs)
    plt.imshow(true_im2)
    plt.title('True image')
    plt.colorbar()

    plt.subplot(2, ncol, 4 + offs)
    plt.imshow(pred_im2)
    plt.title('Predicted image')
    plt.colorbar()

    if camera_im is not None:
        plt.subplot(2, ncol, 5 + offs)
        plt.imshow(camera_im)
        plt.title('Camera image')
        plt.colorbar()

    if pausetime is not None:
        plt.pause(pausetime)


def load_prepared_data():
    """
    Loads, normalises, and splits the data once.
    Cached so Bayesian optimisation does not re-read the NPZ files every trial.
    """
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    metadata = {}

    # Load PL data
    plsavefname = load_precombined_PLims_filename + '.npz'
    print('Loading pl images from ' + plsavefname)
    npf = np.load(datadir + plsavefname, allow_pickle=True)
    all_plims = npf['all_plims']
    if do_subset_on_read and use_subset is not None:
        all_plims = all_plims[:use_subset, :, :]
    all_slmims_filenames = npf['all_slmims_filenames']

    # Load PSF data
    if precombined_psf_filename is None:
        psfsavefname = load_precombined_PLims_filename + '-PSFs' + '.npz'
    else:
        psfsavefname = precombined_psf_filename + '.npz'
    print('Loading psf images from ' + psfsavefname)
    npf = np.load(datadir + psfsavefname, allow_pickle=True)
    all_psfims = npf['all_psfims']
    if do_subset_on_read and use_subset is not None:
        all_psfims = all_psfims[:use_subset, :, :]
    datafilename = plsavefname

    # Load WF data
    wfsavefname = load_precombined_wfims_filename + '.npz'
    print('Loading wf images from ' + wfsavefname)
    npf = np.load(slmdatadir + wfsavefname, allow_pickle=True)
    all_pupphase = npf['all_pupphase']
    if do_subset_on_read and use_subset is not None:
        all_pupphase = all_pupphase[:use_subset, :, :]
    slmloc = npf['slmloc']

    # Apply subset
    if use_subset is not None and do_subset_on_read is False:
        all_plims = all_plims[:use_subset, :, :]
        all_pupphase = all_pupphase[:use_subset, :, :]
        all_psfims = all_psfims[:use_subset, :, :]

    # Normalise
    normfacts = {}

    mn = np.mean(all_plims[:stat_frms])
    all_plims = all_plims - mn
    sd = np.std(all_plims[:stat_frms])
    all_plims = all_plims / sd
    normfacts['PL'] = np.array([mn, 0, sd])

    mn = np.mean(all_pupphase[:stat_frms])
    all_pupphase = all_pupphase - mn
    sd = np.std(all_pupphase[:stat_frms])
    all_pupphase = all_pupphase / sd
    normfacts['WF'] = np.array([mn, 0, sd])

    psf_mn = np.percentile(all_psfims[:stat_frms], 0.1)
    all_psfims = all_psfims - psf_mn
    psf_mx = np.percentile(all_psfims[:stat_frms], 99.9)
    all_psfims = all_psfims / psf_mx
    normfacts['PSF'] = np.array([psf_mn, psf_mx])

    Xdata = all_plims
    ydata_wf = all_pupphase
    ydata_psf = all_psfims

    if addnoise_PL is not None:
        Xdata = Xdata + np.random.normal(0, addnoise_PL, Xdata.shape)
    if addnoise_PSF is not None:
        ydata_psf = ydata_psf + np.random.normal(0, addnoise_PSF, ydata_psf.shape)
    if addnoise_WF is not None:
        ydata_wf = ydata_wf + np.random.normal(0, addnoise_WF, ydata_wf.shape)

    # Split train/test exactly like your original code
    ndata = Xdata.shape[0]
    n_testdata = int(ndata * testdatasplit)
    splitinds = [0, int(n_testdata)]

    X_test = Xdata[splitinds[0]:splitinds[1], :]
    y_test_wf = ydata_wf[splitinds[0]:splitinds[1], :]
    y_test_psf = ydata_psf[splitinds[0]:splitinds[1], :]

    traindata_X_1 = Xdata[:splitinds[0], :]
    traindata_y_wf_1 = ydata_wf[:splitinds[0], :]
    traindata_y_psf_1 = ydata_psf[:splitinds[0], :]

    traindata_X_2 = Xdata[splitinds[1]:, :]
    traindata_y_wf_2 = ydata_wf[splitinds[1]:, :]
    traindata_y_psf_2 = ydata_psf[splitinds[1]:, :]

    X_train = np.vstack((traindata_X_1, traindata_X_2))
    y_train_wf = np.vstack((traindata_y_wf_1, traindata_y_wf_2))
    y_train_psf = np.vstack((traindata_y_psf_1, traindata_y_psf_2))

    _DATA_CACHE = {
        "X_train": X_train,
        "y_train_wf": y_train_wf,
        "y_train_psf": y_train_psf,
        "X_test": X_test,
        "y_test_wf": y_test_wf,
        "y_test_psf": y_test_psf,
        "normfacts": normfacts,
        "metadata": metadata,
        "datafilename": datafilename,
    }
    return _DATA_CACHE


def build_model(pdict, Xndims, yndims_psf, yndims_wf):
    input_img = Input(shape=(Xndims[0], Xndims[1], 1))

    # Encoder
    model_enc = Conv2D(pdict['nfilts_enc'], (pdict['ksz_enc'], pdict['ksz_enc']),
                       activation=pdict['actFunc'], padding='same')(input_img)
    model_enc = Dropout(pdict['dropout_rate'])(model_enc)
    model_enc = MaxPooling2D((2, 2), padding='same')(model_enc)

    model_enc = Conv2D(pdict['nfilts_enc'], (pdict['ksz_enc'], pdict['ksz_enc']),
                       activation=pdict['actFunc'], padding='same')(model_enc)
    model_enc = Dropout(pdict['dropout_rate'])(model_enc)
    model_enc = MaxPooling2D((2, 2), padding='same')(model_enc)

    model_enc = Conv2D(pdict['nfilts_enc'], (pdict['ksz_enc'], pdict['ksz_enc']),
                       activation=pdict['actFunc'], padding='same')(model_enc)
    model_enc = Dropout(pdict['dropout_rate'])(model_enc)
    model_enc = MaxPooling2D((2, 2), padding='same')(model_enc)

    model_enc = Conv2D(pdict['nfilts_enc'], (pdict['ksz_enc'], pdict['ksz_enc']),
                       activation=pdict['actFunc'], padding='same')(model_enc)
    model_enc = Dropout(pdict['dropout_rate'])(model_enc)
    model_enc = MaxPooling2D((2, 2), padding='same')(model_enc)

    # Bottleneck
    model_enc = Conv2D(pdict['nfilts_enc'], (pdict['ksz_enc'], pdict['ksz_enc']),
                       activation=pdict['actFunc'], padding='same', name='Bottleneck')(model_enc)

    # Dense after encoding
    if pdict['n_units_dense'] > 0:
        post_enc_shape = model_enc.shape[1:]
        post_dense_shape = post_enc_shape
        model_enc = Flatten()(model_enc)
        model_enc = Dense(pdict['n_units_dense'], activation=pdict['actFunc'])(model_enc)
        model_enc = Dropout(pdict['dropout_rate_dense'])(model_enc)
        model_enc = Dense(pdict['n_units_dense'], activation=pdict['actFunc'])(model_enc)
        model_enc = Dropout(pdict['dropout_rate_dense'])(model_enc)
        model_enc = Dense(pdict['n_units_dense'], activation=pdict['actFunc'])(model_enc)
        model_enc = Dense(int(np.prod(post_dense_shape)), activation=pdict['actFunc'])(model_enc)
        model_enc = Reshape(post_dense_shape)(model_enc)

    # PSF decoder
    model_psf = Conv2D(pdict['nfilts_psf'], (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation=pdict['actFunc'], padding='same')(model_enc)
    model_psf = Conv2D(pdict['nfilts_psf'], (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation=pdict['actFunc'], padding='same')(model_psf)
    model_psf = Dropout(pdict['dropout_rate_psf'])(model_psf)
    model_psf = UpSampling2D((2, 2), interpolation='bilinear')(model_psf)

    model_psf = Conv2D(pdict['nfilts_psf'], (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation=pdict['actFunc'], padding='same')(model_psf)
    model_psf = Conv2D(pdict['nfilts_psf'], (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation=pdict['actFunc'], padding='same')(model_psf)
    model_psf = Dropout(pdict['dropout_rate_psf'])(model_psf)
    model_psf = UpSampling2D((2, 2), interpolation='bilinear')(model_psf)

    model_psf = Conv2D(pdict['nfilts_psf'], (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation=pdict['actFunc'], padding='same')(model_psf)
    model_psf = Conv2D(pdict['nfilts_psf'], (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation=pdict['actFunc'], padding='same')(model_psf)
    model_psf = Dropout(pdict['dropout_rate_psf'])(model_psf)
    model_psf = UpSampling2D((2, 2), interpolation='bilinear')(model_psf)

    model_psf = Conv2D(pdict['nfilts_psf'], (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation=pdict['actFunc'], padding='same')(model_psf)
    model_psf = Dropout(pdict['dropout_rate_psf'])(model_psf)
    model_psf = Conv2D(pdict['nfilts_psf'], (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation=pdict['actFunc'], padding='same')(model_psf)
    model_psf = Resizing(yndims_psf[0], yndims_psf[1], interpolation='bilinear')(model_psf)
    model_psf = Conv2D(pdict['nfilts_psf'], (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation=pdict['actFunc'], padding='same')(model_psf)
    model_psf = Conv2D(1, (pdict['ksz_psf'], pdict['ksz_psf']),
                       activation='linear', padding='same', name='outlayer_psf')(model_psf)

    # WF decoder
    model_wf = Conv2D(pdict['nfilts_wf'], (pdict['ksz_wf'], pdict['ksz_wf']),
                      activation=pdict['actFunc'], padding='same')(model_enc)
    model_wf = Dropout(pdict['dropout_rate'])(model_wf)
    model_wf = UpSampling2D((2, 2), interpolation='bilinear')(model_wf)

    model_wf = Conv2D(pdict['nfilts_wf'], (pdict['ksz_wf'], pdict['ksz_wf']),
                      activation=pdict['actFunc'], padding='same')(model_wf)
    model_wf = UpSampling2D((2, 2), interpolation='bilinear')(model_wf)

    model_wf = Conv2D(pdict['nfilts_wf'], (pdict['ksz_wf'], pdict['ksz_wf']),
                      activation=pdict['actFunc'], padding='same')(model_wf)
    model_wf = Dropout(pdict['dropout_rate'])(model_wf)
    model_wf = UpSampling2D((2, 2), interpolation='bilinear')(model_wf)

    model_wf = Conv2D(pdict['nfilts_wf'], (pdict['ksz_wf'], pdict['ksz_wf']),
                      activation=pdict['actFunc'], padding='same')(model_wf)
    model_wf = Dropout(pdict['dropout_rate'])(model_wf)
    model_wf = Conv2D(1, (pdict['ksz_wf'], pdict['ksz_wf']),
                      activation='linear', padding='same', name='outlayer_wf')(model_wf)

    model = Model(inputs=input_img, outputs=[model_psf, model_wf])

    callbacks = []
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=pdict['reduceLR_factor'],
        patience=pdict['reduceLR_patience'],
        min_lr=pdict['reduceLR_min_lr'],
        cooldown=pdict['reduceLR_cooldown']
    )

    if pdict['enable_lr_sched']:
        callbacks.append(reduce_lr)
        pdict['learningRate'] = pdict['reduceLR_start']

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=pdict['learningRate']),
        loss=[pdict['lossFunc_psf'], pdict['lossFunc_wf']],
        loss_weights=[pdict['loss_weight'], 1.0]
    )

    return model, callbacks


def train_one_run(
    pdict_override= None, # for optimiser run params, for regular = none
    do_predictions=False,
    do_plotting=False,
    save_model=False,
    save_preds=False,
    save_movie=False,
    verbose=0,
):
    pdict = get_base_pdict()
    if pdict_override is not None:
        pdict.update(pdict_override)

    # Make sure integer-type params stay integers
    pdict['batchSize'] = int(pdict['batchSize'])
    pdict['ksz_enc'] = int(pdict['ksz_enc'])
    pdict['ksz_psf'] = int(pdict['ksz_psf'])
    pdict['ksz_wf'] = int(pdict['ksz_wf'])
    pdict['nfilts_enc'] = int(pdict['nfilts_enc'])
    pdict['nfilts_psf'] = int(pdict['nfilts_psf'])
    pdict['nfilts_wf'] = int(pdict['nfilts_wf'])
    pdict['n_units_dense'] = int(pdict['n_units_dense'])

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    data = load_prepared_data()
    X_train = data["X_train"]
    y_train_wf = data["y_train_wf"]
    y_train_psf = data["y_train_psf"]
    X_test = data["X_test"]
    y_test_wf = data["y_test_wf"]
    y_test_psf = data["y_test_psf"]

    Xndims = X_train.shape[1:]
    yndims_psf = y_train_psf.shape[1:]
    yndims_wf = y_train_wf.shape[1:]

    model, callbacks = build_model(pdict, Xndims, yndims_psf, yndims_wf)

    t = time.time()
    history = model.fit(
        X_train,
        [y_train_psf, y_train_wf],
        validation_data=(X_test, [y_test_psf, y_test_wf]),
        epochs=pdict['epochs'],
        batch_size=pdict['batchSize'],
        callbacks=callbacks,
        verbose=verbose
    )
    print('Total time: %.2f seconds' % (time.time() - t))

    history_loss = history.history['loss']
    history_val_loss = history.history['val_loss']

    # This is the scalar the Bayesian optimiser should minimise
    objective_val_loss = float(np.min(history_val_loss))
    final_val_loss = float(history_val_loss[-1])

    predictions_psf = None
    predictions_wf = None

    if do_predictions or do_plotting or save_preds:
        predictions = model.predict(X_test[:num_preds, :, :], verbose=0)
        predictions_psf = predictions[0]
        predictions_wf = predictions[1]

    if do_plotting:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

        loss_path = datadir + save_filename_pref + timestamp + "_loss-curve.png"
        movie_path = datadir + save_filename_pref + timestamp + "_preds-movie.mp4"

        lossfig = plt.figure(1)
        plt.clf()
        plt.plot(history_loss)
        plt.plot(history_val_loss)
        plt.title('Model loss - val=%.3g' % history_val_loss[-1])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        lossfig.savefig(loss_path, dpi=300, bbox_inches="tight")
        print("Saved loss curve to:", loss_path)
        plt.show()

        num_testims = 10
        wf_true = y_test_wf[:num_testims, :, :]
        wf_pred = predictions_wf[:num_testims, :, :]
        psf_true = y_test_psf[:num_testims, :, :]
        psf_pred = predictions_psf[:num_testims, :, :]
        cur_psf_true_og = None

        fig = plt.figure(2, figsize=(9, 8))

        if save_movie:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='pl'), bitrate=1800)
            writer.setup(fig, movie_path, dpi=100)
            print("Saving movie to:", movie_path)

        for k in np.arange(0, num_testims, 1):
            plot_truepredims(
                wf_true[k, :, :], wf_pred[k, :, :],
                psf_true[k, :, :], psf_pred[k, :, :],
                camera_im=cur_psf_true_og, pausetime=1
            )
            if save_movie:
                writer.grab_frame()

        if save_movie:
            writer.finish()
            print("Saved movie to:", movie_path)

    return {
        "objective_val_loss": objective_val_loss,
        "final_val_loss": final_val_loss,
        "history_val_loss": history_val_loss,
        "history_loss": history_loss,
        "pdict": copy.deepcopy(pdict),
        "model": model,
    }


if __name__ == "__main__":
    result = train_one_run(
        pdict_override=None,
        do_predictions=True,
        do_plotting=True,
        save_model=True,
        save_preds=True,
        save_movie=True,
        verbose=1,
    )
    print("Best val_loss seen in run:", result["objective_val_loss"])
    print("Final val_loss:", result["final_val_loss"])
