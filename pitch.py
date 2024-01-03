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
import tkinter as tk 
from tkinter import filedialog

audiopath = None
recordpath = None

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

  space2.config(text=int(alignment.normalizedDistance))
  #score = 'dtw normalized distance of mfcc: ' + str(alignment.normalizedDistance)
  #print('dtw normalized distance of mfcc: ', alignment.normalizedDistance) #數值越小越相似

def load_vocal(audiopath: str, y, sr: int):
  #path could be designed to make it more organized
  separator_path = os.path.splitext(os.path.basename(audiopath))[0]
  vocal_path = separator_path + '/vocals.wav'
  if not os.path.isfile(vocal_path): #if cwd/[filename]/vocal.wav doesn't exist
    separator = Separator("spleeter:2stems")#拆分原音檔 分成人聲與伴奏
    vocals = separator.separate(waveform=y)['vocals'] #return: dictionary ['vocals':array(),...].
    #waveform.shape=(frames, channels)
    #array.shape = (frames, channels)
    os.mkdir(separator_path)
    sf.write(file=vocal_path, data=vocals, samplerate=sr)
  else:
    vocals,_ = sf.read(vocal_path)
  #vocals.shape = (frames, channels)
  return vocals #low efficiency I think

#從以下两個function找到音檔的路徑 
def upload_file1():
    path=filedialog.askopenfilename()
    global audiopath
    if path:
        audiopath=path
        print(f"Uploading Music from file: {audiopath}") 

def upload_file2():
    path=filedialog.askopenfilename()
    global recordpath
    if path:
        recordpath=path
        print(f"Uploading Music from file: {recordpath}") 

def main():
  #audiopath = args.audioFilename
  #recordpath = args.recordFilename
  y1, sr = librosa.load(audiopath, sr=None, mono=False) #load audio file
  y1 = y1.transpose()
  y2,_ = librosa.load(recordpath, sr=sr, mono=False) #load record file
  y2 = y2.transpose()
  # y.shape = (frames, channels)

  vocals = load_vocal(audiopath=audiopath, y=y1, sr=sr)
  
  #stereo to mono by cauculating mean of all channels
  if (len(vocals.shape) == 2):
    vocals = vocals.mean(axis=1)
  if (len(y2.shape) == 2):
    y2 = y2.mean(axis=1)
  # simply append to same length for dtw
  if (vocals.shape[0] >= y2.shape[0]) :
     y2 = np.insert(arr=y2, obj=y2.shape[0], values=[0 for i in range(vocals.shape[0]-y2.shape[0])])
  else:
     vocals = np.insert(arr=vocals, obj=vocals.shape[0], values=[0 for i in range(y2.shape[0]-vocals.shape[0])])
  #print('shape:', vocals.shape, y2.shape)
     
  compare_similarity(vocals, y2, sr=sr)

  '''
  consume too much memory in dtw
  pitch_record = pitch_detection(y2, sr=sr).selected_array['frequency']
  pitch_vocal = pitch_detection(vocals, sr=sr).selected_array['frequency'] #maybe pitch shift for better compare
  print('dtw normalized distance of pitch', dtw(pitch_vocal, pitch_vocal).normalizedDistance)
  space3.config(text=dtw(pitch_vocal, pitch_record).normalizedDistance)
  '''

  return

'''
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

'''
if __name__ == "__main__":

# 視窗設定
  window=tk.Tk() 
  window.title("評分系統")
  window.minsize(width=400,height=600)# 最小視窗大小
  window.resizable(False,False)
  window.configure(background='#3f4040')# 視窗背景顏色

#排版
  label=tk.Label(window,text="grading system",font=('Arial',24),bg='#3f4040',fg='white')
  space1=tk.Label(window,text=" ",font=('Arial',18),bg='#3f4040',fg='#4F4F4F',justify='left')
  space2=tk.Label(window,text=" ",font=('Arial',48),bg='#3f4040',fg='white',justify='left')
  space3=tk.Label(window,text=" ",font=('Arial',10),bg='#3f4040',fg='#4F4F4F',justify='left')

  space5=tk.Label(window,text=" ",font=('Arial',18),bg='#3f4040',fg='#4F4F4F',justify='left')
  space1.grid(row=0 , column=0,sticky='w',padx = 20,pady = 25)

  label.grid(row=1 , column=1,sticky='w',padx = 20,pady = 5)
  space2.grid(row=2 , column=1,sticky='w',padx = 20,pady = 25)
  space3.grid(row=3 , column=0,sticky='w',padx = 20,pady = 25)

  space5.grid(row=8 , column=0,sticky='w',padx = 20,pady = 25)

  original_muise_button = tk.Button(window,background="#f0a4ad",fg="white",font=('Arial',14), text="original muise",bd=0,width=25,height=2,borderwidth =0,highlightthickness=0,command=lambda: upload_file1())
  your_voice_button = tk.Button(window,background="#9d9fa0",fg="white",font=('Arial',14), text="your voice",bd=0,width=25,height=2,borderwidth =0,highlightthickness=0,command=lambda: upload_file2())
  start_button = tk.Button(window,background="#d84b5b",fg="white",font=('Arial',14), text="start",bd=0,width=25,height=4,borderwidth =0,highlightthickness=0,command=main)
  #button 排版
  original_muise_button.grid(row= 5, column=1,sticky='w',padx = 20,pady = 2)
  your_voice_button.grid(row=6 , column=1,sticky='w',padx=20,pady = 2)
  start_button.grid(row=7 , column=1,sticky='w',padx=20,pady = 9)
  window.mainloop()
