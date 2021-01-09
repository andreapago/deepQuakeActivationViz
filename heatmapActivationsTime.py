from dailystream import DailyStream, get_stations
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops
#import keract
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pickle
import cv2
from tensorflow.keras import backend as K
import matplotlib.pylab as pylab



def load_model():
    """Load the CNN model for the given feature

    Parameters
    ----------
    feature_name : `str`
        Name of the feature to be loaded (`"detection"`, `"azimuth"`,
        `"depth"`, `"distance"`, `"magnitude"`).

    Returns
    -------
    `Model`
        A TensorFlow model of the neural network.
    """
    print(_DATA_STREAM_LENGTH)
    model_input0 = layers.Input(shape=(61, 35, 3))
    model_input1 = layers.Input(shape=(_DATA_STREAM_LENGTH, 3))
    model_input2 = layers.Input(shape=())
    model_output = detection_NN_architecture(model_input1)
    model = Model(inputs=model_input1, outputs=model_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model




def detection_NN_architecture(Tinputs):
    feature_name = "detection"
    x = layers.Conv1D(64, activation='relu', batch_input_shape=(None, _DATA_STREAM_LENGTH, 3),
                      data_format='channels_last', kernel_size=10, strides=4,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(Tinputs)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # #x = layers.Dropout(0.1)(x)
    # x = layers.Conv1D(64, activation='relu', kernel_size=10, strides=2,
    #                   kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # #x = layers.Dropout(0.1)(x)
    # x = layers.Conv1D(32, activation='relu', kernel_size=5, strides=2,
    #                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    # x = layers.Dropout(0.25)(x)
    # x = layers.Conv1D(32, activation='elu', kernel_size=3, strides=2,
    #                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001), padding='valid')(x)
    x = layers.Flatten()(x)
    last = layers.Dense(64, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(x)
    output = layers.Dense(3,
                          activation='softmax', name=(feature_name + '_out'))(last)

    return output



def load_pickles_expl(pickleFilenameAmp, pickleFilenameTime):
    amplitude_data = pickle.load(open(pickleFilenameAmp, 'rb'))
    start_times = pickle.load(open(pickleFilenameTime, 'rb'))

    loaded_data = {
        'amplitude_data': amplitude_data,
        'start_times': start_times,
        #'latitude': lat,
        #'longitude': lon
    }

    return loaded_data



def cam(model,input_data,ev_type):
    input_dataM = np.expand_dims(input_data, axis=0)
    preds = model.predict(input_dataM)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    layer_names = [layer.name for layer in model.layers]
    for layer in layer_names:
        if "conv" in layer:
            conv_layer = model.get_layer(layer)
            iterate = K.function([model.input], [model.output, conv_layer.output[0]])
            mod_out, conv_layer_output_value = iterate([input_dataM])
            for i in range(64):
                heatmap =conv_layer_output_value[:,i]
                #print(heatmap)
                heatmap = cv2.resize(heatmap, (3, 2000))
                heatmap = np.uint8(255 * heatmap)
                plot_data(input_data, heatmap,i,layer,ev_type)
    return 0



def plot_colourline(x,y,c):
    import matplotlib.cm as cm
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])
    norm = mpl.colors.Normalize(vmin=np.min(c), vmax=np.max(c))
    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    return ax,c

def plot_data(stream_data, heatmap,filterNo,layerName,ev_type):#, data_info, output_dir=None):
    """Plots the stream amplitudes as stored in the numpy array."""

    params = {'legend.fontsize': '22',
              # 'figure.figsize': (15, 5),
              'axes.labelsize': '22',
              'axes.titlesize': '22',
              'xtick.labelsize': '22',
              'ytick.labelsize': '22'}
    pylab.rcParams.update(params)

    fig =plt.figure(figsize=(5, 5), dpi=200)
    num_channels = stream_data.shape[1]
    a = range(0, 2000)

    fig = plt.figure()
    fig.set_size_inches(10, 15)

    ax1 = fig.add_subplot(311)
    channel = stream_data[:, 0]
    plot_colourline(a, channel, heatmap[:, 0])
    #plt.colorbar()

    ax2 = fig.add_subplot(312)
    channel = stream_data[:, 1]
    plot_colourline(a, channel, heatmap[:, 1])
    #plt.colorbar()

    ax3 = fig.add_subplot(313)
    channel = stream_data[:, 2]
    plot_colourline(a, channel, heatmap[:, 2])
    #plt.colorbar()

    ax1.title.set_text('Samples channel E')

    ax2.title.set_text('Samples channel N')
    ax3.title.set_text('Samples channel Z')
    ax3.set_xlabel("Sample number in window")
    ax3.set_ylabel("Normalized Amplitude")







    # for i in range(num_channels):
    #     channel = stream_data[:, i]
    #     plt.subplot(num_channels, 1, i + 1)
    #
    #     a= range(0,2000)
    #     plot_colourline(a,channel, heatmap[:,i])
    #     if i==0:
    #         plt.title.set_text('Samples channel E')
    #     if i==1:
    #         plt.title.set_text('Samples channel N')
    #     if i==2:
    #         plt.title.set_text('Samples channel Z')
    #         plt.set_xlabel("Sample number in window")
    #         plt.set_ylabel("Normalized Amplitude")
    from pathlib import Path
    Path("timeActivations/"+ev_type+"/"+layerName).mkdir(parents=True, exist_ok=True)
    plt.savefig(("timeActivations/"+ev_type+"/"+layerName+"/time_seism_ev_"+layerName+"_"+str(filterNo)+".png"))
    plt.close(fig)











#using a day with a seismic event and noise
pickle_dir = "data/time/"
station = "G623"
day = 142
sample_info = DailyStream.unpickle_samples(pickle_dir, station, day, load_frequencies=False)

input_data = sample_info['amplitude_data']
start_times = sample_info['start_times']
latitude = sample_info['latitude']
longitude = sample_info['longitude']

_DATA_STREAM_LENGTH = 2000


model = load_model()
model.load_weights("weights/detection_saved_wt_best-MODEL_time.h5")

#plot of a noise example
heatmapPropNoise = cam(model,input_data[0][1000],"noise")
#plot of an event 1 type example
heatmapPropEvent1 = cam(model,input_data[0][1374],"seis_ev")


#using a sample of acoustic event, type 2 event
sample_info = load_pickles_expl("data/acoustic_sample_data.pkl", "data/acoustic_sample_times.pkl")
input_data = sample_info['amplitude_data']

#for the plot of an event 2 type
heatmapProp = cam(model,input_data[0][0])