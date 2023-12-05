import numpy as np
import librosa
import sounddevice as sd
from spleeter.separator import Separator
#from spleeter.audio.adapter import AudioAdapter
#import matplotlib.pyplot as plt
import soundfile as sf
import os
import sys
import argparse
import asyncio
from dtw import dtw
import parselmouth

#path = "./audio.wav"

def pitch_detection(waveform: np.ndarray)->parselmouth.Pitch:
  snd = parselmouth.Sound(values=waveform, sampling_frequency=sr)
  pitch = snd.to_pitch()
  return pitch

def compare_similarity(vocal, record):
  #Mel-Frequency Cepstral Coefficients
  mfcc1 = librosa.feature.mfcc(y=vocal, sr=sr)
  mfcc2 = librosa.feature.mfcc(y=record, sr=sr)

  #資料不能太大 這個DP的時間複雜度好像是exponential?
  alignment = dtw(mfcc1, mfcc2) #time series data之間的距離

  print('dtw normalized distance: ', alignment.normalizedDistance) #數值越小越相似

async def record(buffer: np.ndarray, sr) : #參考sounddevice sample
  loop = asyncio.get_event_loop()
  event = asyncio.Event()
  idx = 0

  def record_callback(indata: np.ndarray, frames: int, time, status) -> None:
    nonlocal idx
    if status:
      print(status, file=sys.stderr)
    remainder = len(buffer) - idx
    if remainder == 0:
      loop.call_soon_threadsafe(event.set)
      raise sd.CallbackStop
    indata = indata [:remainder]
    buffer[idx:idx+len(indata)] = indata
    idx += len(indata)
  
  print('start recording')
  in_stream = sd.InputStream(samplerate=sr, channels=2, callback=record_callback)
  with in_stream:
    await event.wait()

async def main():
  
  # y.shape = (frames, channels)
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
  '''
  pitch = pitch_detection(vocal_path)
  '''
  #start recording along with playing music
  buffer = np.empty(y.shape)
  sd.play(data=y, samplerate=sr) #data.shape = (frames, channels)
  await record(buffer=buffer, sr=sr)
  sd.wait()
  print('recording end')
  
  #stereo to mono by cauculating mean of all channels
  vocals = vocals.mean(axis=1)
  buffer = buffer.mean(axis=1)
  print('shape:', vocals.shape, buffer.shape)
  compare_similarity(vocals, buffer)

  pitch_record = pitch_detection(buffer).to_array()
  pitch_vocal = pitch_detection(vocals).to_array()
  print(dtw(pitch_vocal, pitch_vocal))
  # real-time ploting is under developing
  #anim = FuncAnimation(fig=plt.gcf(), func=record_animation, frames=buffer.shape[0], interval=30)
  #anim = FuncAnimation(fig=plt.gcf(), func=plot_animation, frame=pitch_detected.shape[0], interval=300)
  #plt.show()

parser = argparse.ArgumentParser('pitch.py', add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio device')
args, remaining = parser.parse_known_args()
if args.list_devices:
  print(sd.query_devices())
  parser.exit(0)

parser.add_argument('audioFilename', metavar='AUDIOFILENAME', help='chosen audio file')
parser.add_argument('recordFilename', metavar='RECORDFILENAME', help='chosen correspond record file')
args = parser.parse_args(remaining)

if __name__ == "__main__":
  audiopath = args.audioFilename
  recordpath = args.recordFilename
  y, sr = librosa.load(audiopath, sr=None, mono=False) #load audio file
  y = y.transpose()
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    sys.exit('\nInterrupt by user')
