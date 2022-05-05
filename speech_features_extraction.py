import os
from glob import glob
from shutil import rmtree
from argparse import ArgumentParser

from tqdm import tqdm
from config import PATH_TO_FEATURES, PATH_TO_RAWDATA

import numpy as np

import librosa

import pandas as pd

parser = ArgumentParser()
parser.add_argument('--task', type=str, default='humor', choices=['humor', 'reaction', 'stress'])
parser.add_argument('--toolkit', type=str, default='SMILExtract', help='used tool kit')
parser.add_argument('--feature', type=str, default='is09', help='feature set name')
parser.add_argument('--frame_size', type=float, default=2.0, help='frame size')
parser.add_argument('--frame_step', type=float, default=0.5, help='frame step')
parser.add_argument('--feature_set_config', type=str, default='../opensmile/config/is09-13/IS09_emotion.conf', help='Specify the features set config used (only one).')

def main(args):
    toolkit = args.toolkit
    frame_size = args.frame_size
    frame_step = args.frame_step
    feature_set_config = args.feature_set_config
    target_dir = os.path.join(PATH_TO_FEATURES[args.task], f'{args.feature}-{frame_size}-{frame_step}')

    if os.path.exists(target_dir):
        rmtree(target_dir)
    os.makedirs(target_dir)

    wavs = sorted(glob(os.path.join(PATH_TO_RAWDATA[args.task], 'audio', '**', '*.wav'), recursive=True))
    # wavs = ['../MuSe2022/c1_muse_humor/raw_data/audio/baum/baum_01/baum_01_1116500_1133500.wav', '../MuSe2022/c1_muse_humor/raw_data/audio/baum/baum_01/baum_01_146000_163000.wav']

    first_line_meta = ['timestamp', 'segment_id']
    first_line_feature = [('feature_' + str(i)) for i in range(1, 385)]
    first_line = ','.join(first_line_meta) + ',' + ','.join(first_line_feature) + '\n'

    for wav in tqdm(wavs):
        last_part = wav.replace(os.path.join(PATH_TO_RAWDATA[args.task], 'audio'), "")
        last_part = last_part.replace(".wav" ,".csv")
        last_part = last_part.split('/')
        filename = last_part[-1]
        last_part = last_part[:-2]
        last_part.append(filename)
        last_part = '/'.join(last_part)
        target_csv = target_dir + last_part
        os.makedirs(os.path.dirname(target_csv), exist_ok=True)
        instname = filename.replace(".csv", "")

        duration = librosa.get_duration(filename=wav)
        for start in np.arange(0, duration, frame_step):
            end = min(start + frame_step, duration)
            command = '{} -C {} -I {} -start {} -end {} -headercsvlld 0 -instname {} -O {} 2>/dev/null'.format(toolkit, feature_set_config, wav, start, end, instname, target_csv)
            os.system(command)

        with open(target_csv, 'r') as f:
            lines_tmp = f.readlines()
            lines_tmp = lines_tmp[391:]
            f.close()

        lines = []
        lines.append(first_line)
        timestamp = 0
        for feature_line in lines_tmp:
            line = str(timestamp) + ',' + feature_line
            line = line[:-3] + '\n'
            lines.append(line)
            timestamp += int(args.frame_step * 1000)

        with open(target_csv, 'w') as f:
            f.writelines(lines)
            f.close()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
