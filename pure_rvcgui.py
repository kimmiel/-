import tkinter as tk
from tkinter import filedialog
import os

myfont = ('Arial', 15)

class Rvc_gui(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(Menu)
        
        self.configure(bg='#20330E')
        self.title("Retrieval-based Voice Conversion")
        self.geometry("500x700")
        self.resizable(False,False)
        #self.iconbitmap("")        # icon setting
        
    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack(expand=True, fill='both')          # cover fullscreen
    
    #def pack()
 
class Menu(tk.Frame):       
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg="lightblue")
        tk.Label(self, text="Welcome to RVC", font=('Arial', 30)).place(relx=0.5,rely=0.1,anchor='center')
        tk.Button(self, text="train your own model!!!", font=myfont, relief=tk.GROOVE, command=lambda: master.switch_frame(Train_page)
            ).place(relx=0.5,rely=0.4,anchor='center')         # train_btn
        tk.Button(self, text="get your cover!!!", font=myfont, relief=tk.GROOVE
            ).place(relx=0.5,rely=0.6,anchor='center')         # cover_btn
        tk.Button(self, text="seperates vocal from your source!!!", font=myfont, relief=tk.GROOVE
            ).place(relx=0.5,rely=0.8,anchor='center')         # sep_btn


class Train_page(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='purple')
        tk.Label(self, text="train_page", font=('Arial', 30)).place(relx=0.5,rely=0.1,anchor='center')
        tk.Label(self, text="repo name?", font=myfont).place(relx=0.2,rely=0.25,anchor='center')
        self.repo_entry = tk.Entry(self, width=20, font=('Arial', 18))
        self.repo_entry.place(relx=0.6,rely=0.25,anchor='center', width=250, height=30)
        tk.Label(self, text="repo name?", font=myfont).place(relx=0.2,rely=0.25,anchor='center')
        tk.Label(self, text="voice directory?", font=myfont).place(relx=0.2,rely=0.4,anchor='center')
        self.select_btn = tk.Button(self, text="select", font=myfont, command=self.choose_dir)
        self.select_btn.place(relx=0.4,rely=0.4,anchor='w')
        tk.Label(self, text="total epoch?", font=myfont).place(relx=0.2,rely=0.55,anchor='center')
        self.epoch_entry = tk.Entry(self, width=20, font=('Arial', 18))
        self.epoch_entry.place(relx=0.6,rely=0.55,anchor='center', width=250, height=30)

        tk.Button(self, text="one click to train", font=('Arial',30), command=lambda: self.combined_func(master)
                ).place(relx=0.5,rely=0.8,anchor='center')
        tk.Button(self, text="back", font=myfont, command=lambda: master.switch_frame(Menu)
                ).place(relx=0,rely=0,anchor='nw')

    
    
    def choose_dir(self):
        path = filedialog.askdirectory(title='select a directory')
        if path:
            self.song_dir = path
            last_dir = os.path.basename(path)
            self.select_btn.configure(text='...\%s' %last_dir, font=('Arial',12))
            
    
    def combined_func(self, master):
        self.repo_name = self.repo_entry.get()
        self.epoch = self.epoch_entry.get()
        master.switch_frame(Processing_page)
        master.update()
        self.start_training(master)
            
        
        
    def start_training(self, master):               # 先把功能註解掉，方便做ui
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
        master.switch_frame(Processing_page2)

            
            
            


class Processing_page(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='black')
        tk.Label(self, text="Processing...", font=('Arial', 30), bg='black',fg='white').place(relx=0.2,rely=0.4,anchor='w')
        #### 可能可以做個動畫
        self.done_btn = tk.Button(self, text="Done!", font=('Arial',20), bg='white',fg='red', command=lambda: master.switch_frame(Menu)
                ).place_forget()

class Processing_page2(Processing_page):
    def __init__(self, master):
        Processing_page.__init__(self, master)
        self.done_btn = tk.Button(self, text="Done!", font=('Arial',20), bg='white',fg='red', command=lambda: master.switch_frame(Menu)
            ).place(relx=0.5,rely=0.8,anchor='center')



window = Rvc_gui()

window.mainloop()