import csv
from distutils.command.config import config
import librosa
import os
import numpy as np
import config
from glob import glob
from tqdm import tqdm

def save_SS(path,feats):
    feats=np.squeeze(feats)
    np.save(path,feats)

def _get_this_SS(mfccs):
    '''
    for mfccs of this clip
    :param mfccs:
    :return:
    '''
    D=[]
    for i in range(len(mfccs)-1):
        d_i=mfccs[i]
        d_j=mfccs[i+1]
        D.append(0.5*(d_i+d_j))
    return D

def _get_SS(wave):
    '''
    get Spectral Stability for this wave
    :param wave:
    :return:
    '''
    segment_id = str(wave.split('\\')[-1][:-4])
    SS_feats=[]
    frame_step=0.5
    duration=librosa.get_duration(filename=wave)
    for start in np.arange(0, duration, frame_step):
        if start+frame_step < duration:
            durat=frame_step
        else:
            durat=duration-start
        y,sr=librosa.load(wave,offset=start,duration=durat)
        mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40) #return [40,a]
        SS=_get_this_SS(mfccs.T) #input:[a,40]
        #SS shape:[20,40]
        SS_feats.append([int(start*1000),SS])
    return SS_feats

def RawDataLoader():
    wavs = sorted(glob(os.path.join(config.PATH_TO_RAWDATA['humor'], 'audio', '**', '*.wav'), recursive=True))
    return wavs

def main():
    wavs=RawDataLoader()
    path_to_SS_feats=os.path.join(config.PATH_TO_FEATURES['humor'],'spectral_stability')
    if not os.path.exists(path_to_SS_feats):
        os.makedirs(path_to_SS_feats)
    for wav in tqdm(wavs):
        coach_id=wav.split('\\')[-3]
        filename=wav.split('\\')[-1][:-4]+".npy"
        this_path=os.path.join(path_to_SS_feats,coach_id)
        if not os.path.exists(this_path):
            os.makedirs(this_path)
        save_path=os.path.join(this_path,filename)
        SS_feats=_get_SS(wav)
        save_SS(save_path,SS_feats)
    print('done')



if __name__ == '__main__':
    main()
    