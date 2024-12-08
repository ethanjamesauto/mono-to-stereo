import sklearn.model_selection
import utils
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
import sklearn
import numpy as np
import scipy.io.wavfile
from glob import glob
import os

stft = utils.get_stft(fs=utils.FS)

features = []
labels = []

n_files = 10 # use all files
files = glob('./data/**/**.wav')[:n_files]

np.random.seed(0)
np.random.shuffle(files)

# use train_test_split instead
train, test = sklearn.model_selection.train_test_split(files, test_size=0.2)

total_frames = 3000 # the paper uses half a million frames
n_frames = 20 # the paper uses 20 frames per feature
frames_per_song = total_frames // len(train)

print(f'train: {len(train)}, test: {len(test)}, frames_per_song: {frames_per_song}')

for f in tqdm(train):
    if f.find('_hat.wav') > 0: # TODO: really need a more reliable way of doing this
        continue
   
    s, BS, P_IID, P_IC, P, m = utils.file_to_feature_label(f, stft)

    for i in range(frames_per_song):
        # take a random set of frames from the spectrogram S
        idx = np.random.randint(0, BS.shape[1] - n_frames)

        feature = BS[:, idx:idx+n_frames]
        feature = utils.complex_to_real(feature).flatten()
        features.append(feature)
        
        # the label is the PS parameters of the last frame
        label = P[:, idx+n_frames]
        label = utils.complex_to_real(label)
        labels.append(label)

knn = KNeighborsRegressor(n_neighbors=1).fit(features, labels)

# save knn
import pickle
with open('knn_' + str(total_frames) + '.pkl', 'wb') as f:
    pickle.dump(knn, f)

# also save a csv with information about the model
import pandas as pd
df = pd.DataFrame({
    'n_files': n_files,
    'total_frames': total_frames,
    'n_frames': n_frames,
    'frames_per_song': frames_per_song,
    'time': pd.Timestamp.now()
}, index=[0])
df.to_csv('knn_' + str(total_frames) + '.csv', index=False)
