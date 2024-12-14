import sklearn.model_selection
import utils
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
import sklearn
import numpy as np
import scipy.io.wavfile
from glob import glob
import os
from bokeh.plotting import figure
from bokeh.models import LogColorMapper, ColorBar
import bokeh.palettes as pl
import panel as pn

# delete all files with _hat.wav
for f in glob('data/**/**_hat.wav'):
    os.remove(f)

stft = utils.get_stft(fs=utils.FS)

total_frames = 500000 # the paper uses half a million frames

# open metadata csv named 'knn_' + str(total_frames) + '.csv'
import pandas as pd
metadata = pd.read_csv('knn_' + str(total_frames) + '.csv')


n_files = metadata['n_files'][0] # use all files
files = glob('./data/**/**.wav')[:n_files]

np.random.seed(0)
np.random.shuffle(files)

# use train_test_split instead
train, test = sklearn.model_selection.train_test_split(files, test_size=0.2)

n_frames = 20 # the paper uses 20 frames per feature


# load knn
import pickle
with open('knn_' + str(total_frames) + '.pkl', 'rb') as f:
    knn = pickle.load(f)


n_test = len(test)

panel = pn.Column()

for l in [train[:n_test], test[:n_test]]:
    # test the model
    mean_mse_IID = 0
    mean_mse_IC = 0 

    mean_mse_IID_baseline = 0
    mean_mse_IC_baseline = 0

    for f in tqdm(l):
        print(f)
        s, BS, P_IID, P_IC, P, m = utils.file_to_feature_label(f, stft)
        P_hat = np.zeros_like(P)

        features = []
        for i in range(BS.shape[1] - n_frames):
            feature = BS[:, i:i+n_frames]
            feature = utils.complex_to_real(feature).flatten()
            features.append(feature)
        
        labels = knn.predict(features)
        for i in range(len(labels)):
            P_hat[:, i+n_frames] = utils.real_to_complex(labels[i])
        
        P_IID_hat, P_IC_hat = utils.parameters_split(P_hat)
        l_hat, r_hat = utils.decode(stft=stft, s=s, P_IID=P_IID_hat, P_IC=P_IC_hat)
        scipy.io.wavfile.write(f.replace('.wav', '_hat.wav'), utils.FS, np.array([l_hat*m, r_hat*m]).astype(np.int16).T)

        # save the original file using PS parameters as _orig_hat.wav
        l, r = utils.decode(stft=stft, s=s, P_IID=P_IID, P_IC=P_IC)
        scipy.io.wavfile.write(f.replace('.wav', '_orig_hat.wav'), utils.FS, np.array([l*m, r*m]).astype(np.int16).T)

        # print mse
        # P_IID /= np.linalg.norm(P_IID)
        # P_IC /= np.linalg.norm(P_IC)
        # P_IID_hat /= np.linalg.norm(P_IID_hat)
        # P_IC_hat /= np.linalg.norm(P_IC_hat)

        mse_IID = np.mean(np.abs(P_IID - P_IID_hat)**2)
        mse_IC = np.mean(np.abs(P_IC - P_IC_hat)**2)

        mse_IID_baseline = np.mean(np.abs(P_IID)**2)
        mse_IC_baseline = np.mean(np.abs(P_IC)**2)

        mean_mse_IID += mse_IID/n_test
        mean_mse_IC += mse_IC/n_test

        mean_mse_IID_baseline += mse_IID_baseline/n_test
        mean_mse_IC_baseline += mse_IC_baseline/n_test

        print(f'mse_IID={mse_IID:.2f}, mse_IC={mse_IC:.2f}')
        print(f'mse_IID_baseline={mse_IID_baseline:.2f}, mse_IC_baseline={mse_IC_baseline:.2f}')


        for img in [np.abs(P_IID), np.abs(P_IID_hat), np.abs(P_IC), np.abs(P_IC_hat)]:
            img = img[:, :500]
            m = LogColorMapper(palette=pl.Inferno256, low=1e-5, high=img.max())
            p = figure(width=1500, height=700, title='Spectrogram', x_axis_label='Time (s)', y_axis_label='Frequency (kHz)')
            p.min_border=0

            p.image(image=[img], x=0, y=0, dw=30, dh=44.1/2, color_mapper=m)
            p.add_layout(ColorBar(color_mapper=m), 'right')
            panel.append(p)

    print(f'mean_mse_IID={mean_mse_IID:.2f}, mean_mse_IC={mean_mse_IC:.2f}')
    print(f'mean_mse_IID_baseline={mean_mse_IID_baseline:.2f}, mean_mse_IC_baseline={mean_mse_IC_baseline:.2f}')

panel.save('spectrogram.html')

