try:
    import tensorflow as tf
    import numpy as np
    import time
    import matplotlib.pyplot as plt
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Flatten, Dense, Reshape, Dropout, MaxPooling2D, Resizing
    
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

except ImportError:
    print('Warning - COULD NOT IMPORT TENSORFLOW')
import numpy as np
import platform
import matplotlib
os_name = platform.system()
if 'Linux' in os_name:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import time
import datetime

# Data directories
datadir = '/home/manav//PL-NN-testdata_forDec2025/'
# datadir = '/home/bnorris/Data/PL/20240605_labdata_subset/'
slmdatadir = datadir
outdir = datadir

# File naming
save_filename_pref = 'pl2wf2psf_data202407_model01_'
outname_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")
save_filename = save_filename_pref + outname_datetime

# Precombined data filenames
load_precombined_PLims_filename = 'pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined'
precombined_psf_filename = None  # None for auto
load_precombined_wfims_filename = 'slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined'

"""
Experiments to consider
Train on clean data, predict on noisy PL data (how do hparams affect this)
Train on noisy PL data, predict on noisy PL data (maybe this is better)
Train on noisy data (including WF & PSF)
"""

# Perturbation experiments - add noise
addnoise_PL = None #0.1 # Noise in units of sigma (=None to skip noise adding)
addnoise_PSF = None #0.1 # Noise in units of sigma (=None to skip noise adding)
addnoise_WF = None #0.1 # Noise in units of sigma (=None to skip noise adding)

# Image processing parameters
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

# Training parameters
keep_orig_psfs = False
stat_frms = 1000# 10000  # How many frames to use for normalisation statistics
testdatasplit = 0.2
shuffle_before_split = False  # Should be false for time-correlated data
use_subset = 10000 # set to None to use all frames
num_preds = 100  # -1 for all
do_subset_on_read = False

# Output settings
save_model = True
save_preds = True
doplotting = True
save_movie = True #True

# Model parameters
pdict = {}
pdict['actFunc'] = 'relu'
pdict['batchSize'] = 32
pdict['learningRate'] = 0.00011219438754796004
pdict['lossFunc_psf'] = 'mean_squared_error'
pdict['lossFunc_wf'] = 'mean_squared_error'
pdict['epochs'] = 50
pdict['dropout_rate'] = 0.1
pdict['dropout_rate_dense'] = 0.3
pdict['dropout_rate_psf'] = 0.3
pdict['ksz_enc'] = 5
pdict['ksz_psf'] = 3
pdict['ksz_wf'] = 5
pdict['nfilts_enc'] = 96
pdict['nfilts_psf'] = 64
pdict['nfilts_wf'] = 64
pdict['loss_weight'] = 1  # >1 to weight PSF higher
pdict['n_units_dense'] = 4096
pdict['enable_lr_sched'] = False
pdict['reduceLR_start'] = 1e-4
pdict['reduceLR_factor'] = 0.5
pdict['reduceLR_patience'] = 5
pdict['reduceLR_min_lr'] = 1e-6
pdict['reduceLR_cooldown'] = 5

# Load PL data
metadata = {}
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

# Apply subset if needed
if use_subset is not None and do_subset_on_read is False:
    all_plims = all_plims[:use_subset, :, :]
    all_pupphase = all_pupphase[:use_subset, :, :]
    all_psfims = all_psfims[:use_subset, :, :]

# Normalize PL input data
normfacts = {}
mn = np.mean(all_plims[:stat_frms])
all_plims -= mn
sd = np.std(all_plims[:stat_frms])
all_plims /= sd
normfacts['PL'] = np.array([mn, 0, sd])

# Normalize WF images
mn = np.mean(all_pupphase[:stat_frms])
all_pupphase -= mn
sd = np.std(all_pupphase[:stat_frms])
all_pupphase /= sd
normfacts['WF'] = np.array([mn, 0, sd])

# Normalize PSF images
psf_mn = np.percentile(all_psfims[:stat_frms], 0.1)
all_psfims -= psf_mn
psf_mx = np.percentile(all_psfims[:stat_frms], 99.9)
all_psfims /= psf_mx
normfacts['PSF'] = np.array([psf_mn, psf_mx])

Xdata = all_plims
ydata_wf = all_pupphase
ydata_psf = all_psfims

if addnoise_PL is not None:
    Xdata += np.random.normal(0, addnoise_PL, Xdata.shape)
if addnoise_PSF is not None:
    ydata_psf += np.random.normal(0, addnoise_PSF, ydata_psf.shape)
if addnoise_WF is not None:
    ydata_wf += np.random.normal(0, addnoise_WF, ydata_wf.shape)


# Split train test
ndata = Xdata.shape[0]
res_testframes = ndata * testdatasplit
n_testdata = int(res_testframes)
n_traindata = int(ndata - n_testdata)
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
del ydata_wf, ydata_psf, Xdata
del traindata_X_1, traindata_y_wf_1, traindata_y_psf_1, traindata_X_2, traindata_y_wf_2, traindata_y_psf_2

print(pdict)
X_val, y_val_psf, y_val_wf = X_test, y_test_psf, y_test_wf

# ---- Add channel dimension so Conv2D input/output shapes match ----
X_train = X_train[..., None]
X_test  = X_test[..., None]
X_val   = X_val[..., None]

y_train_psf = y_train_psf[..., None]
y_test_psf  = y_test_psf[..., None]
y_val_psf   = y_val_psf[..., None]

y_train_wf = y_train_wf[..., None]
y_test_wf  = y_test_wf[..., None]
y_val_wf   = y_val_wf[..., None]


# Function to plot true vs predicted images
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
    plt.subplot(2, ncol, 3+offs)
    plt.imshow(true_im2)
    plt.title('True image')
    plt.colorbar()
    plt.subplot(2, ncol, 4+offs)
    plt.imshow(pred_im2)
    plt.title('Predicted image')
    plt.colorbar()
    if camera_im is not None:
        plt.subplot(2, ncol, 5+offs)
        plt.imshow(camera_im)
        plt.title('Camera image')
        plt.colorbar()
    if pausetime is not None:
        plt.pause(pausetime)

# Build the neural network model
Xndims = X_train.shape[1:]
yndims_psf = y_train_psf.shape[1:]
yndims_wf = y_train_wf.shape[1:]
input_img = Input(shape=(Xndims[0], Xndims[1], 1))

# Encoder
def build_model(pdict, Xndims, yndims_psf, yndims_wf):
    # IMPORTANT: create Input INSIDE the function (fresh graph each trial)
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
        post_enc_shape = tuple(int(d) for d in model_enc.shape[1:])
        reconstruct_units = int(np.prod(post_enc_shape))

        model_enc = Flatten()(model_enc)
        model_enc = Dense(int(pdict['n_units_dense']), activation=pdict['actFunc'])(model_enc)
        model_enc = Dropout(pdict['dropout_rate_dense'])(model_enc)
        model_enc = Dense(int(pdict['n_units_dense']), activation=pdict['actFunc'])(model_enc)
        model_enc = Dropout(pdict['dropout_rate_dense'])(model_enc)
        model_enc = Dense(int(pdict['n_units_dense']), activation=pdict['actFunc'])(model_enc)
        model_enc = Dense(reconstruct_units, activation=pdict['actFunc'])(model_enc)
        model_enc = Reshape(post_enc_shape)(model_enc)

    # PSF Decoder
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

    # WF Decoder
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

    model = keras.Model(inputs=input_img, outputs=[model_psf, model_wf])
    return model


# Create the model with two outputs
# model = Model(inputs=input_img, outputs=[model_psf, model_wf])
#model.summary()

# skopt: tune ONLY learningRate, dropout_rate, n_units_dense
# ----------------------------
base_pdict = dict(pdict)

space = [
    Real(3e-4, 3e-3, prior="log-uniform", name="learningRate"),
    Real(0.0, 0.5, name="dropout_rate"),
    Integer(2048, 4096, name="n_units_dense"),
]

TUNE_EPOCHS = 5

@use_named_args(space)
def objective(learningRate, dropout_rate, n_units_dense):
    trial = dict(base_pdict)
    trial["learningRate"] = float(learningRate)
    trial["dropout_rate"] = float(dropout_rate)
    trial["n_units_dense"] = int(n_units_dense)

    # If you ever enable lr scheduling, be aware it can override the tuned LR.
    if trial.get("enable_lr_sched", False):
        trial["learningRate"] = trial["reduceLR_start"]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    model = build_model(trial, Xndims, yndims_psf, yndims_wf)

    # callbacks (keep your ReduceLROnPlateau behaviour if enabled + add EarlyStopping for tuning)
    callbacks = []
    if trial.get("enable_lr_sched", False):
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=trial['reduceLR_factor'],
                patience=trial['reduceLR_patience'],
                min_lr=trial['reduceLR_min_lr'],
                cooldown=trial['reduceLR_cooldown']
            )
        )

    callbacks.append(
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=trial["learningRate"]),
        loss=[trial["lossFunc_psf"], trial["lossFunc_wf"]],
        loss_weights=[trial["loss_weight"], 1.0],
    )

    history = model.fit(
        X_train, [y_train_psf, y_train_wf],
        validation_data=(X_val, [y_val_psf, y_val_wf]),
        epochs=TUNE_EPOCHS,
        batch_size=trial["batchSize"],
        shuffle=False,
        verbose=0,
        callbacks=callbacks
    )

    best_val = float(np.min(history.history["val_loss"]))
    return best_val


res = gp_minimize(
    objective,
    dimensions=space,
    n_calls=15,
    n_initial_points=5,
    random_state=42,
)

best_params = {dim.name: val for dim, val in zip(space, res.x)}
print("Best val_loss:", res.fun)
print("Best params:", best_params)


# ----------------------------
# Train FINAL model with best params (then continue as your script)
# ----------------------------
final_pdict = dict(base_pdict)
final_pdict.update(best_params)

model = build_model(final_pdict, Xndims, yndims_psf, yndims_wf)
model.summary()

callbacks = []
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=final_pdict['reduceLR_factor'],
    patience=final_pdict['reduceLR_patience'],
    min_lr=final_pdict['reduceLR_min_lr'],
    cooldown=final_pdict['reduceLR_cooldown']
)
if final_pdict['enable_lr_sched']:
    callbacks.append(reduce_lr)
    final_pdict['learningRate'] = final_pdict['reduceLR_start']

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=final_pdict['learningRate']),
    loss=[final_pdict['lossFunc_psf'], final_pdict['lossFunc_wf']],
    loss_weights=[final_pdict['loss_weight'], 1.0]
)

t = time.time()
history = model.fit(
    X_train, [y_train_psf, y_train_wf],
    validation_data=(X_val, [y_val_psf, y_val_wf]),
    epochs=final_pdict['epochs'],
    batch_size=final_pdict['batchSize'],
    callbacks=callbacks,
    shuffle=False,
    verbose=1
)
print('Total time: %.2f seconds' % (time.time() - t))

del X_train, y_train_psf, y_train_wf

history_loss = history.history['loss']
history_val_loss = history.history['val_loss']

# Save the model after training is complete
if save_model:
    modelfilename = datadir + save_filename + '.h5'
    modelfilename_nopath = save_filename + '.h5'
    print('Saving model to ' + modelfilename)
    model.save(modelfilename, save_format='h5')
    np.savez(datadir + save_filename + '_metadata.npz', pdict=pdict,
             pldata_filename=datafilename, loss=history_loss,
             val_loss=history_val_loss, metadata=metadata, normfacts=normfacts)
else:
    modelfilename_nopath = ''

# Make predictions
predictions = model.predict(X_test[:num_preds, :, :])
predictions_psf = predictions[0]
predictions_wf = predictions[1]

# Save predictions
if save_preds:
    y_test_psf_og_save = None
    predfilename = datadir + save_filename + '_preds.npz'
    predfilename_nopath = save_filename + '_preds.npz'
    print('Saving predictions to ' + predfilename)
    np.savez(predfilename, predvals_psf=predictions_psf, predvals_wf=predictions_wf,
             testvals_psf=y_test_psf[:num_preds, :, :], testvals_wf=y_test_wf[:num_preds, :, :],
             history_loss=history_loss, history_val_loss=history_val_loss,
             X_test=X_test[:num_preds, :, :], pdict=pdict, y_test_psf_og=y_test_psf_og_save,
             pldata_filename=datafilename, normfacts=normfacts)
else:
    predfilename_nopath = ''

# Plot results if requested
if doplotting:
    if history_loss is not None:
        # Plot training & validation loss values
        plt.figure(1)
        plt.clf()
        plt.plot(history_loss)
        plt.plot(history_val_loss)
        plt.title('Model loss - val=%.3g' % history_val_loss[-1])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        

    # Look at results
    num_testims = 10#00
    wf_true = y_test_wf[:num_testims, :, :]
    wf_pred = predictions_wf[:num_testims, :, :]
    psf_true = y_test_psf[:num_testims, :, :]
    psf_pred = predictions_psf[:num_testims, :, :]
    cur_psf_true_og = None
    
    # Plot image results
    imshow_stride = 1
    pausetime = 1#0.001
    
    fig = plt.figure(2, figsize=(9, 8))
    if save_movie:
        moviefilename = datadir + save_filename + '_preds-movie.mp4'
        import matplotlib.animation as animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='pl'), bitrate=1800)
        writer.setup(fig, moviefilename, dpi=100)

    for k in np.arange(0, num_testims, imshow_stride):
        plot_truepredims(wf_true[k, :, :], wf_pred[k, :, :],
                         psf_true[k, :, :], psf_pred[k, :, :],
                         camera_im=cur_psf_true_og, pausetime=pausetime)
        if save_movie:
            writer.grab_frame()
    if save_movie:
        writer.finish()
