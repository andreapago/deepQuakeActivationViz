from dailystream import DailyStream, get_stations
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras import backend as K
import cv2
import matplotlib.cm as cm
import pickle
from scipy import signal
import matplotlib.pylab as pylab
from pathlib import Path


_DATA_STREAM_LENGTH = 2000



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
    model_input0 = layers.Input(shape=(61, 35, 3))
    model_input1 = layers.Input(shape=(_DATA_STREAM_LENGTH, 3))
    model_input2 = layers.Input(shape=())
    model_output = detection_NN_architecture_freq_only(model_input0)
    model = Model(inputs=model_input0, outputs=model_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model



def detection_NN_architecture_freq_only(Sinputs):
    feature_name = "detection"

    y = layers.Conv2D(64, activation='relu', batch_input_shape=(None, (61, 35), 3),
                      data_format='channels_last', kernel_size=(2, 2), strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001))(Sinputs)

    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)

    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=2,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)

    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)
    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)
    y = layers.Conv2D(64, activation='relu', kernel_size=(2, 2), strides=1,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), padding='valid')(y)

    y = layers.Flatten()(y)

    output = layers.Dense(3,
                          activation='softmax', name=(feature_name + '_out'))(y)


    return output


def cam(model,input_data,evType):
    #input_dataM = np.expand_dims(input_data, axis=0)
    #input_dataM = np.reshape(input_dataM, (1, 61, 35, 3))
    channel1 = input_data[0][2] #frequencies
    channel2 = input_data[1][2]
    channel3 = input_data[2][2]
    timeScale = input_data[0][1]
    freqScale = input_data[0][0]

    A = np.empty((1, 61, 35, 3))
    A[0, :, :, 0] = channel1
    A[0, :, :, 1] = channel2
    A[0, :, :, 2] = channel3

    preds = model.predict(A)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    layer_names = [layer.name for layer in model.layers]
    for layer in layer_names:
        if "conv" in layer:
            conv_layer = model.get_layer(layer)
            iterate = K.function([model.input], [model.output, conv_layer.output[0]])
            mod_out, conv_layer_output_value = iterate([A])
            for i in range(64):
                heatmap =conv_layer_output_value[:,:,i]
                #print(heatmap)
                heatmap = cv2.resize(heatmap, (61,35))
                heatmap = np.uint8(255 * heatmap)
                plot_data(A, timeScale, freqScale, heatmap,i,layer,evType)
    return 0



def plot_data(stream_data, timeScale, freqScale, heatmap,filterNo,layerName,evType):#, data_info, output_dir=None):
    """Plots the stream amplitudes as stored in the numpy array."""


    params = {'legend.fontsize': '22',
              #'figure.figsize': (15, 5),
              'axes.labelsize': '22',
              'axes.titlesize': '22',
              'xtick.labelsize': '22',
              'ytick.labelsize': '22'}
    pylab.rcParams.update(params)

    timeScale = 20 * (timeScale / max(timeScale))  # rescaling to 20 sec
    freqScale = 50 * (freqScale / max(freqScale))  # rescaling to 0-50Hz

    channelData0 = stream_data[0, :, :, 0]
    channelData1 = stream_data[0, :, :, 1]
    channelData2 = stream_data[0, :, :, 2]

    fig = plt.figure()
    fig.set_size_inches(10, 15)

    ax1 = fig.add_subplot(411)
    plt.pcolormesh(timeScale, freqScale, channelData0, vmin=0, vmax=np.max(channelData0), shading='gouraud')
    plt.colorbar()
    ax2 = fig.add_subplot(412)
    plt.pcolormesh(timeScale, freqScale, channelData1, vmin=0, vmax=np.max(channelData1), shading='gouraud')
    plt.colorbar()
    ax3 = fig.add_subplot(413)
    plt.pcolormesh(timeScale, freqScale, channelData2, vmin=0, vmax=np.max(channelData2), shading='gouraud')
    plt.colorbar()
    ax4 = fig.add_subplot(414)
    heatmap2 = np.flip(heatmap, axis=0)
    if (np.max(heatmap2)!=0):
        heatmap2 = (heatmap2/np.max(heatmap2))
    plt.imshow(heatmap2, interpolation='nearest', aspect="auto")
    plt.colorbar()
    plt.axis('off')
    ax1.title.set_text('Spectrogram channel E')
    ax2.title.set_text('Spectrogram channel N')
    ax3.title.set_text('Spectrogram channel Z')
    ax4.title.set_text('Activation heatmap')
    fig.tight_layout(pad=0.25)
    #plt.imshow()



    Path("frequencyActivations/"+evType+"/"+layerName).mkdir(parents=True, exist_ok=True)
    plt.savefig(("frequencyActivations/"+evType+"/"+layerName+"/frequency_"+layerName+"_"+str(filterNo)+".png"))
    plt.close(fig)





model = load_model()
model.load_weights("weights/detection_saved_wt_best-MODEL_freq_only.h5")


#code for noise and seismic events
##########################################

pickle_dir = "data/freq"
station = "G623"
day = 142

sample_info = DailyStream.unpickle_samples(pickle_dir, station, day,
                                           load_frequencies=True)

#input_data = sample_info['amplitude_data']
start_times = sample_info['start_times']
latitude = sample_info['latitude']
longitude = sample_info['longitude']
input_data = sample_info['frequency_data']


#noise sample
heatmapPropFreq = cam(model,input_data[1000],"noise")
#seismic event sample
heatmapPropFreq = cam(model,input_data[1374],"seis_ev")
##########################################



#below code for running the acoustic sample

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



sample_info = load_pickles_expl("data/acoustic_sample_data.pkl", "data/acoustic_sample_times.pkl")
sample = sample_info["amplitude_data"][0][0]
freq_data_acoustic = []
for trace in range(0,3):
    f, t, zxx = signal.stft(sample[:,trace], window='hanning', nperseg=120)
    freq_data_acoustic.append([f, t, np.abs(zxx)])
heatmapFreqAcoustic = cam(model,freq_data_acoustic,"acou_ev")