import numpy as np
import librosa
import sounddevice as sd
from spleeter.separator import Separator
import soundfile as sf
import os
import sys
import argparse
from dtw import dtw
import parselmouth

#path = "./audio.wav"

def pitch_detection(waveform: np.ndarray, sr)->parselmouth.Pitch:
  snd = parselmouth.Sound(values=waveform, sampling_frequency=sr)
  pitch = snd.to_pitch()
  return pitch

def compare_similarity(vocal, record, sr):
  #Mel-Frequency Cepstral Coefficients
  mfcc1 = librosa.feature.mfcc(y=vocal, sr=sr)
  mfcc2 = librosa.feature.mfcc(y=record, sr=sr)

  #資料不能太大 這個DP的時間複雜度好像是exponential?
  alignment = dtw(mfcc1, mfcc2) #time series data之間的距離

  print('dtw normalized distance of mfcc: ', alignment.normalizedDistance) #數值越小越相似

def main():
  audiopath = args.audioFilename
  recordpath = args.recordFilename
  y1, sr = librosa.load(audiopath, sr=None, mono=False) #load audio file
  y1 = y1.transpose()
  y2 = librosa.load(audiopath, sr=sr, mono=False) #load record file
  y2 = y2.transpose()
  # y.shape = (frames, channels)

  #path could be designed to make it more organized
  separator_path = os.path.splitext(os.path.basename(audiopath))[0]
  vocal_path = separator_path + '/vocals.wav'
  if not os.path.isfile(vocal_path): #if cwd/[filename]/vocal.wav doesn't exist
    separator = Separator("spleeter:2stems")#拆分原音檔 分成人聲與伴奏
    vocals = separator.separate(waveform=y1)['vocals'] #return: dictionary ['vocals':array(),...].
    #waveform.shape=(frames, channels)
    #array.shape = (frames, channels)
    os.mkdir(separator_path)
    sf.write(file=vocal_path, data=vocals, samplerate=sr)
  else:
   vocals,_ = sf.read(vocal_path)
  #vocals.shape = (frames, channels)
  
  #stereo to mono by cauculating mean of all channels
  vocals = vocals.mean(axis=1)
  y2 = y2.mean(axis=1)
  #print('shape:', vocals.shape, buffer.shape)
  compare_similarity(vocals, y2, sr=sr)

  pitch_record = pitch_detection(y2).to_array()
  pitch_vocal = pitch_detection(vocals).to_array()
  print('dtw normalized distance of pitch', dtw(pitch_vocal, pitch_vocal).normalizedDistance)

  return 0


#command line argument parser
parser = argparse.ArgumentParser('pitch.py', add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio device')
args, remaining = parser.parse_known_args()
if args.list_devices:
  print(sd.query_devices())
  parser.exit(0)

parser.add_argument('audioFilename', metavar='AUDIOFILENAME', help='chosen audio file')
parser.add_argument('recordFilename', metavar='RECORDFILENAME', help='chosen correspond record file')
args = parser.parse_args(remaining)

#print dtw normalized distance可以改成寫到檔案
if __name__ == "__main__":
  main()
