rvc(使用github開源專案 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

環境配置:
    1. 如果要有辦法執行rvc.py，並成功跳出gui
        建議先建置一個虛擬環境(python>=3.8)，
        再執行pip install -r requirements.txt(電腦是n卡)，可能要等他跑一下

        A卡/I卡：
        pip install -r requirements-dml.txt

        A卡Rocm（Linux）：
        pip install -r requirements-amd.txt

        I卡IPEX（Linux）：
        pip install -r requirements-ipex.txt

        其他配置可以參考 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI



    2. 如果真的想訓練模型，
        先試試抓不抓的到顯卡，寫一段python
        {import torch
            torch.cuda.is_available() 
            如果是true代表抓的到，如果是false，參考以下文件
            https://zhuanlan.zhihu.com/p/409616444
            https://medium.com/@anannannan0102/pytorch-%E6%83%B3%E7%94%A8gpu%E8%B7%91ml%E7%92%B0%E5%A2%83%E5%8D%BB%E8%A3%9D%E4%B8%8D%E5%A5%BD-%E7%9C%8B%E5%AE%8C%E9%80%99%E7%AF%87%E5%B8%B6%E4%BD%A0%E9%81%BF%E9%96%8B%E5%90%84%E7%A8%AE%E9%9B%B7-3bf259fc7396
            建議配置: python3.9, cuda12.1, torch 2.1.1+cu121(去官網載)
            其他版本要自己試一下。
        }

        如果抓不到顯卡，用cpu應該也行，但是會跑比較慢


程式運行hint:
    功能一. 訓練模型
    1.  repository_name 代表實驗資料夾名稱，可自訂
        填寫實驗配置. 實驗資料放在logs下, 每個實驗一個資料夾, 內含實驗配置, 日誌, 訓練得到的模型文件
    2.  voice_directory 選擇訓練文件夾，裡面最好只放一個10min以上音檔
    3.  total_epoch 訓練總輪數，視音檔品質決定，100-200基本就夠了

    ->訓練完的.pth 會在assets/weights/, .index在logs/repository_name/

    功能二. 推理模型
    1.  vocal directory 代表想推理的歌曲所放的資料夾
    2.  transposition 表示變調，幾個半音
    3.  choose index跟pth可以透過scroll來瀏覽

    ->推理完，可以在cover資料夾內取得推理歌聲

    功能三. vocal分離
    1.  song directory代表目標歌曲所在資料夾
    ->完成後，vocal檔會在vocal資料夾，而instrumental會在instrumental資料夾找到

