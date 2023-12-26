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
                  ).place(relx=0.2, rely=0.4, anchor='w')  # train_btn
        tk.Button(self, text="get your cover!!!", font=myfont, relief=tk.RAISED
                  ).place(relx=0.2, rely=0.6, anchor='w')  # cover_btn
        tk.Button(self, text="seperates vocal from your source!!!", font=myfont, relief=tk.RAISED
                  ).place(relx=0.2, rely=0.8, anchor='w')  # sep_btn


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
            self.song_dir = path
            last_dir = os.path.basename(path)
            self.select_btn.configure(text='...\%s' % last_dir, font=('Arial', 12))

    def combined_func(self, master):
        self.repo_name = self.repo_entry.get()
        self.epoch = self.epoch_entry.get()
        master.switch_frame(Processing_page)
        master.update()
        self.start_training(master)

    def start_training(self, master):  # 先把功能註解掉，方便做ui
        '''
        exp_dir1 = self.repo_name   # repo_name 實驗資料夾名字
        sr2 = "40k"
        if_f0_3 = True
        version19 = "v2"
        np7 =  int(np.ceil(config.n_cpu / 1.5))

        trainset_dir4 = self.song_dir           # 訓練文件夾名字
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
        '''
        time.sleep(3)
        master.switch_frame(Processing_page2)


class Processing_page(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='black')
        self.procLabel = tk.Label(self, text="Processing...", font=('Arial', 30), bg='black', fg='white')
        self.procLabel.place(relx=0.2, rely=0.4, anchor='w')
        #### 可能可以做個動畫
        #self.done_btn = tk.Button(self, text="Done!", font=('Arial', 20), bg='white', fg='red',
        #                          command=lambda: master.switch_frame(Menu)
        #                          ).place_forget()

        
        def update_label():
            # 檢查label是否存在
            if self.procLabel:
                # 取得label文字
                text = self.procLabel.cget("text")

                # 計算句點數量
                dots_count = text.count(".")

                # 如果句點數量等於3，則將文字重置為"Processing"
                if dots_count == 3:
                    self.procLabel.configure(text="Processing")
                else:
                    # 將文字增加一個句點
                    self.procLabel.configure(text=text + ".")
                self.procLabel.place(relx=0.2, rely=0.4, anchor='w')
                # 每隔500毫秒呼叫update_label()函數
                window.after(500, update_label)
            else:
                # label不存在，則退出
                return

        update_label()


class Processing_page2(Processing_page):
    def __init__(self, master):
        Processing_page.__init__(self, master)
        self.done_btn = tk.Button(self, text="Done!", font=('Arial', 20), bg='white', fg='red',
                                  command=lambda: master.switch_frame(Menu)
                                  ).place(relx=0.5, rely=0.8, anchor='center')


window = Rvc_gui()

window.mainloop()
