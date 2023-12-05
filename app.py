
import tkinter as tk
from tkinter import filedialog
import os
import pyaudio
import wave, sys, threading
# import pygame

window=tk.Tk()
window.title("使用AI生成音樂")

window.minsize(width=375,height=667)# 最小視窗大小
window.configure(background='white')# 視窗背景顏色

label=tk.Label(window,text="Using AI to generate your music",font=('Arial',24),bg='white',fg='#4F4F4F',justify='left')
space=tk.Label(window,text=" ",font=('Arial',18),bg='white',fg='#4F4F4F',justify='left')
space.grid(row=0 , column=0,sticky='w',padx = 20,pady = 25)
label.grid(row=1 , column=0,sticky='w',padx = 20,pady = 5)

# the function of upload the musie


def upload_file():
    path=filedialog.askopenfilename()
    if path:
        print(f"Uploading Music from file: {path}")
        # pygame.mixer.init()
        # pygame.mixer.music.load(path)
        # pygame.mixer.music.play()


# 播放
def play_file():
    path = filedialog.askopenfilename()
    if path:
        # use pyaudio to open a stream
        # 讀取 .wav檔
        wf = wave.open(path, 'rb')  # 'rb' : 二進位讀取模式。'wb' : 二進位寫入模式。'r+b' : 二進位讀寫模式。
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        buf = 1024
        while True:
            data = wf.readframes(buf)
            if data == '':
                break
            stream.write(data)

        stream.close()
        p.terminate()

#可以叫出評分頁面的 function
def Scoring():
    # wait where the output  music
    os.system('python3 pitch.py')
    print(f"Scoring function")

#可以叫出RVC 頁面的 function
def RVC_page():
    os.system('python3 請改成RVCpy檔的名字.py')
    print(f"RVC function")

save_button = tk.Button(window,background="#fbabbb",fg="white",font=('Arial',14), text="score",width=50,height=4,borderwidth =0,highlightthickness=0,command=Scoring)
upload_button = tk.Button(window,background="#ea4462",fg="white",font=('Arial',14), text="play muise",width=50,height=4,borderwidth =0,highlightthickness=0,command=play_file)
button3 = tk.Button(window,background="#3e3e3e",fg="white",font=('Arial',14), text="RVC",width=50,height=4,borderwidth =0,highlightthickness=0,command=RVC_page)
button4 = tk.Button(window,background="#fafafa",fg="black",font=('Arial',14), text="備用button4",width=50,height=4,borderwidth =0,highlightthickness=0)
upload_button.grid(row=2 , column=0,sticky='w',padx=20,pady = 5)
save_button.grid(row= 3, column=0,sticky='w',padx = 20,pady = 5)
button3.grid(row=4 , column=0,sticky='w',padx=20,pady = 5)
button4.grid(row=5 , column=0,sticky='w',padx = 20,pady = 5)
window.mainloop()
