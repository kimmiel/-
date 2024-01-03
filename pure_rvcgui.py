import tkinter as tk
from tkinter import filedialog
import os
import time
'''  the following part is for gui  '''
myfont = ('Arial', 15)


class Rvc_gui(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(Menu)

        self.configure(bg='#20330E')
        self.title("Retrieval-based Voice Conversion")
        self.geometry("500x700")
        self.resizable(False, False)
        # self.iconbitmap("")        # icon setting

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack(expand=True, fill='both')  # cover fullscreen

    # def pack()


class Menu(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg="lightblue")
        tk.Label(self, text="Welcome to RVC", font=('Arial', 30), bg='lightblue').place(relx=0.5, rely=0.15, anchor='center')
        tk.Button(self, text="train your own model!!!", font=myfont, relief=tk.RAISED,
                  command=lambda: master.switch_frame(Train_page)
                  ).place(relx=0.2, rely=0.4, anchor='w')  # train_btn (done)
        tk.Button(self, text="get your cover!!!", font=myfont, relief=tk.RAISED,
                  command=lambda: master.switch_frame(Cover_page)
                  ).place(relx=0.2, rely=0.6, anchor='w')  # cover_btn (new)
        tk.Button(self, text="seperates vocal from your source!!!", font=myfont, relief=tk.RAISED,
                  command=lambda: master.switch_frame(UVR_page)
                  ).place(relx=0.2, rely=0.8, anchor='w')  # UVR_btn (new)


class Train_page(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='lightblue')
        tk.Label(self, text="train_page", font=('Arial', 30),  bg='lightblue', fg='white').place(relx=0.5, rely=0.1, anchor='center')
        tk.Label(self, text="repo name?",  bg='lightblue', fg='white', font=myfont).place(relx=0.1, rely=0.25, anchor='w')
        self.repo_entry = tk.Entry(self, width=20, font=('Arial', 18))
        self.repo_entry.place(relx=0.6, rely=0.25, anchor='center', width=250, height=30)
        # tk.Label(self, text="repo name?", font=myfont).place(relx=0.2, rely=0.25, anchor='center')    # 好像多了一次?
        tk.Label(self, text="voice directory?",  bg='lightblue', fg='white', font=myfont).place(relx=0.1, rely=0.4, anchor='w')
        self.select_btn = tk.Button(self, text="select", font=myfont, command=self.choose_dir)
        self.select_btn.place(relx=0.4, rely=0.4, anchor='w')
        tk.Label(self, text="total epoch?",  bg='lightblue', fg='white', font=myfont).place(relx=0.1, rely=0.55, anchor='w')
        self.epoch_entry = tk.Entry(self, width=20, font=('Arial', 18))
        self.epoch_entry.place(relx=0.6, rely=0.55, anchor='center', width=250, height=30)

        tk.Button(self, text="one click to train", font=('Arial', 30), command=lambda: self.combined_func(master)
                  ).place(relx=0.5, rely=0.8, anchor='center')
        tk.Button(self, text="back", font=myfont, command=lambda: master.switch_frame(Menu)
                  ).place(relx=0, rely=0, anchor='nw')
        
    def choose_dir(self):
        path = filedialog.askdirectory(title='select a directory')
        if path:
            self.voice_dir = path
            last_dir = os.path.basename(path)
            self.select_btn.configure(text='...\%s' % last_dir, font=('Arial', 12))

    def combined_func(self, master):
        self.repo_name = self.repo_entry.get()
        self.epoch = self.epoch_entry.get()
        master.switch_frame(Processing_page)
        master.update()
        self.start_training(master)

    def start_training(self, master):  # 先把功能註解掉，方便做ui
        
        exp_dir1 = self.repo_name   # repo_name 實驗資料夾名字
        sr2 = "40k"
        if_f0_3 = True
        version19 = "v2"
        np7 =  int(np.ceil(config.n_cpu / 1.5))

        trainset_dir4 = self.voice_dir           # 訓練文件夾名字
        spk_id5 = 0
        gpus6 = gpus
        gpu_info9 = gpu_info
        f0method8 = "rmvpe"
        gpus_rmvpe ="%s-%s" % (gpus, gpus)

        save_epoch10 = 50
        total_epoch11 = self.epoch               # 總訓練輪數
        batch_size12 = default_batch_size
        if_save_latest13 = ""
        if_cache_gpu17 = ""
        if_save_every_weights18 = ""
        pretrained_G14 = "assets/pretrained_v2/f0G40k.pth"
        pretrained_D15 = "assets/pretrained_v2/f0D40k.pth"
        gpus16 = gpus
        info = " "
        for _ in train1key(exp_dir1,
                    sr2,
                    if_f0_3,
                    trainset_dir4,
                    spk_id5,
                    np7,
                    f0method8,
                    save_epoch10,
                    total_epoch11,
                    batch_size12,
                    if_save_latest13,
                    pretrained_G14,
                    pretrained_D15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                    gpus_rmvpe):
            pass
        
        # time.sleep(1)    # 模擬等待處理效果
        master.switch_frame(Processing_page2)


class UVR_page(tk.Frame):       # ultimate vocal remover
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='purple')
        tk.Label(self, text="UVR_page", font=('Arial', 30), bg='purple', fg='white').place(relx=0.5, rely=0.1, anchor='center')
        tk.Label(self, text="song directory?", font=myfont).place(relx=0.1, rely=0.4, anchor='w')
        self.select_btn = tk.Button(self, text="select", font=myfont, command=self.choose_dir)
        self.select_btn.place(relx=0.5, rely=0.4, anchor='w')
        tk.Button(self, text="one click to remove vocal", font=('Arial', 26), command=lambda: self.combined_func(master)
                  ).place(relx=0.5, rely=0.8, anchor='center')
        tk.Button(self, text="back", font=myfont, command=lambda: master.switch_frame(Menu)
                  ).place(relx=0, rely=0, anchor='nw')
        
        
    def choose_dir(self):
        path = filedialog.askdirectory(title='select a directory')
        if path:
            self.song_dir = path
            last_dir = os.path.basename(path)
            self.select_btn.configure(text='...\%s' % last_dir, font=('Arial', 12))

    
    def combined_func(self, master):
        master.switch_frame(Processing_page)
        master.update()
        self.start_removing(master)
        
        
    def start_removing(self, master):
        dir_wav_input = self.song_dir       # 放要移除人聲的歌曲 的資料夾
        wav_inputs = ""
        model_choose = "HP2"
        opt_vocal_root = "vocal"            # 移除的人聲丟到vocal資料夾
        opt_ins_root = "instrumental"       # 伴奏丟到instrumental
        agg = 10
        format0 = "mp3"
        
        for _ in uvr(model_choose, dir_wav_input, opt_vocal_root, wav_inputs, opt_ins_root, agg, format0):
            pass

        # time.sleep(1)    # 模擬等待處理效果
        master.switch_frame(Processing_page2)


class Cover_page(tk.Frame):         # model referencing(get your cover)
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='purple')
        tk.Label(self, text="cover_page", font=('Arial', 30), bg='purple', fg='white').place(relx=0.5, rely=0.1, anchor='center')
        tk.Label(self, text="vocal directory?", font=myfont).place(relx=0.1, rely=0.25, anchor='w')
        self.select_btn = tk.Button(self, text="select", font=myfont, command=self.choose_dir)
        self.select_btn.place(relx=0.5, rely=0.25, anchor='w')
        
        tk.Label(self, text="transposition(-12~12)?", font=("Arial", 12)).place(relx=0.1, rely=0.35, anchor='w')
        self.trans_entry = tk.Entry(self, width=20, font=('Arial', 18))
        self.trans_entry.place(relx=0.7, rely=0.35, anchor='center', width=250, height=30)


        
        # 重新偵測index
        global index_paths
        index_paths = []
        for root, dirs, files in os.walk(index_root, topdown=False):
            for name in files:
                if name.endswith(".index") and "trained" not in name:
                    index_paths.append("%s/%s" % (root, name))
        
  
        self.id_var = -1         # just for initialization
        self.text_var = tk.StringVar()
        self.text_var.set("")
        tk.Label(self, text="choose index file?", font=myfont).place(relx=0.1, rely=0.45, anchor='w')
        tk.Label(self, textvariable=self.text_var, font=('Arial', 10)).place(relx=0.1, rely=0.5, anchor='w')
        
        self.select_btn2 = tk.Button(self, text="scroll", font=('Arial', 12), command=self.scroll_file)
        self.select_btn2.place(relx=0.5, rely=0.45, anchor='w')

        
        
        # 重新偵測pth
        global names
        names = []
        for name in os.listdir(weight_root):
            if name.endswith(".pth"):
                names.append(name)
        
        self.id_var2 = -1         # just for initialization
        self.text_var2 = tk.StringVar()
        self.text_var2.set("")
        tk.Label(self, text="choose pth file?", font=myfont).place(relx=0.1, rely=0.55, anchor='w')
        tk.Label(self, textvariable=self.text_var2, font=('Arial', 10)).place(relx=0.1, rely=0.6, anchor='w')
        
        self.select_btn3 = tk.Button(self, text="scroll", font=('Arial', 12), command=self.scroll_file2)
        self.select_btn3.place(relx=0.5, rely=0.55, anchor='w')
        
        tk.Button(self, text="get your cover", font=('Arial', 26), command=lambda: self.combined_func(master)
                  ).place(relx=0.5, rely=0.8, anchor='center')
        tk.Button(self, text="back", font=myfont, command=lambda: master.switch_frame(Menu)
                  ).place(relx=0, rely=0, anchor='nw')
    
    def choose_dir(self):
        path = filedialog.askdirectory(title='select a directory')
        if path:
            self.song_dir = path
            last_dir = os.path.basename(path)
            self.select_btn.configure(text='...\%s' % last_dir, font=('Arial', 12))

    def scroll_file(self):          # 捲動index
        self.id_var += 1
        if(self.id_var >= len(index_paths)):
            self.id_var = 0         # refresh
        self.text_var.set(index_paths[self.id_var])
        
    
    def scroll_file2(self):         # 捲動pth
        self.id_var2 += 1
        if(self.id_var2 >= len(names)):
            self.id_var2 = 0         # refresh
        self.text_var2.set(names[self.id_var2])
      
    
    def combined_func(self, master):
        self.trans = self.trans_entry.get()
        master.switch_frame(Processing_page)
        master.update()
        self.start_referencing(master)


    def start_referencing(self, master):    # 開始推理歌聲
        sid0 = self.text_var2.get()           # 選擇使用的pth_file
        spk_item = 0
        vc_transform1 = self.trans       # how many semitones
        opt_input = "cover"                 # 推理出的音頻會丟到cover這個資料夾
        file_index2 = ""
        file_index3 = ""
        file_index4 = self.text_var.get()     # 選擇使用的index_file
        f0method1 = "rmvpe"
        format1 = "mp3"
        resample_sr1 = 0
        rms_mix_rate1 = 1
        protect0 = 0.33
        protect1 = 0.33
        filter_radius1 = 3
        index_rate2 = 0.75              # 特徵占比
        dir_input = self.song_dir       # 原歌資料夾
        inputs = ""

        
        # 重新載入pth
        vc.get_vc(sid0, protect0, protect1)

        
        for _ in vc.vc_multi(spk_item,
                    dir_input,
                    opt_input,
                    inputs,
                    vc_transform1,
                    f0method1,
                    file_index3,
                    file_index4,
                    index_rate2,
                    filter_radius1,
                    resample_sr1,
                    rms_mix_rate1,
                    protect1,
                    format1):
            pass

        # time.sleep(1)    # 模擬等待處理效果
        master.switch_frame(Processing_page2)




class Processing_page(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='black')
        proLabel = tk.Label(self, text="Processing...", font=('Arial', 30), bg='black', fg='white').place(relx=0.2, rely=0.4,
                                                                                               anchor='w')

        #### 可能可以做個動畫
        #self.done_btn = tk.Button(self, text="Done!", font=('Arial', 20), bg='white', fg='red',
        #                          command=lambda: master.switch_frame(Menu)
        #                          ).place_forget()



class Processing_page2(Processing_page):
    def __init__(self, master):
        Processing_page.__init__(self, master)
        self.done_btn = tk.Button(self, text="Done!", font=('Arial', 20), bg='white', fg='red',
                                  command=lambda: master.switch_frame(Menu)
                                  ).place(relx=0.5, rely=0.8, anchor='center')


window = Rvc_gui()

window.mainloop()
