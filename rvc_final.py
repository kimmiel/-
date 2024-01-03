import tkinter as tk
from tkinter import filedialog
import os
import sys
import logging
import shutil
import warnings
import torch
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from configs.config import Config
import fairseq
from time import sleep
from subprocess import Popen
import threading
from random import shuffle
import pathlib
import json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import traceback
import faiss
import time

now_dir = os.getcwd()
sys.path.append(now_dir)

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

config = Config()
vc = VC(config)

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml


# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("很遺憾您這沒有能用的顯示卡來支持您訓練")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])



weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))


sr_dict = {
    #"32k": 32000,
    "40k": 40000,
    #"48k": 48000,
}

def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True



def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    logger.info(cmd)
    # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    p = Popen(cmd, shell=True)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log



# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                    ps.append(p)
                # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log
    # 对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log



def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )




# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
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
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    '''with open('filename.txt', 'w') as file:
        # 在這裡進行文件的操作
        file.write("exp_dir1 : %s\nexp_dir : %s\n" %(exp_dir1, exp_dir))
    '''
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1, # if if_save_latest13 == i18n("是") else 0,
                0, # 1 if if_cache_gpu17 == i18n("是") else 0,
                0, # 1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1,  #if if_save_latest13 == i18n("是") else 0,
                0, # 1 if if_cache_gpu17 == i18n("是") else 0,
                0, # 1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "訓練結束, 您可查看控制台訓練日誌或實驗資料夾下的train.log"



# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "請先進行特徵提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "請先進行特徵提取!"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功建構索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)



# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
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
    gpus_rmvpe,
):
    
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)
    
    # step1:处理数据
    yield get_info_str("step1:正在处理数据")
    [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]
    
    
     # step2a:提取音高
    yield get_info_str("step2:正在提取音高&正在提取特征")
    [
        get_info_str(_)
        for _ in extract_f0_feature(
            gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
        )
    ]

    
   # step3a:训练模型
    yield get_info_str("step3a:正在训练模型")
    click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
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
    )
    yield get_info_str("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log")

    # step3b:训练索引
    [get_info_str(_) for _ in train_index(exp_dir1, version19)]
    yield get_info_str("全流程结束！")
    






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
        tk.Frame.__init__(self, master,bg='lightblue')
        tk.Label(self, text="UVR_page", font=('Arial', 30),bg='lightblue', fg='white').place(relx=0.5, rely=0.1, anchor='center')
        tk.Label(self, text="song directory?",  bg='lightblue', fg='white', font=myfont).place(relx=0.1, rely=0.4, anchor='w')
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
        tk.Frame.__init__(self, master,bg='lightblue')
        tk.Label(self, text="cover_page",   font=('Arial', 30),bg='lightblue', fg='white').place(relx=0.5, rely=0.1, anchor='center')
        tk.Label(self, text="vocal directory?",  bg='lightblue', fg='white', font=myfont).place(relx=0.1, rely=0.25, anchor='w')
        self.select_btn = tk.Button(self, text="select", font=myfont, command=self.choose_dir)
        self.select_btn.place(relx=0.5, rely=0.25, anchor='w')
        
        tk.Label(self, text="transposition(-12~12)?",  bg='lightblue', fg='white', font=("Arial", 12)).place(relx=0.1, rely=0.35, anchor='w')
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
        tk.Label(self, text="choose index file?",  bg='lightblue', fg='white', font=myfont).place(relx=0.1, rely=0.45, anchor='w')
        tk.Label(self, textvariable=self.text_var,  bg='lightblue', fg='white', font=('Arial', 10)).place(relx=0.1, rely=0.5, anchor='w')
        
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
        tk.Label(self, text="choose pth file?",  bg='lightblue', fg='white', font=myfont).place(relx=0.1, rely=0.55, anchor='w')
        tk.Label(self, textvariable=self.text_var2,  bg='lightblue', fg='white', font=('Arial', 10)).place(relx=0.1, rely=0.6, anchor='w')
        
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


