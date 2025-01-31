# GLM-4v-8b-base-recruitment-advertisement

## 微調設備
+ OS: Ubuntu 22.04.2 LTS。
+ Python: 3.10。
+ CUDA Version: 12.1。
+ GPU Driver Version: 550.127.08。
+ GPU: NVIDIA H200-140GB，推理至少需32GB以上的GPU。

## 環境設置
您可以使用以下指令來創建完整 conda 虛擬環境`glm_py_3.10`，將創建到默認的`envs`目錄下:
```bash
conda env create -f environment.yml
```

## 微調模型介紹
`checkpoint-4000_01`為使用玉山提供之資料集`label_0120.xlsx`中前 823 筆資料為訓練集微調，最後 200 筆為測試集，分數如下圖:

![image](https://github.com/sandychinghuang/glm-4v-8b-base-recruitment-advertisement/blob/main/metric.png?raw=true)


## 模型推理
1. 請先使用 `xlsx2jsonl.py` 將包含 file_name 和 result 的 xlsx 檔案轉換成`text.jsonl`。
+ 需更改 user_message 中的 "image"路徑。
+ 需更改 input_file。
2. 若需要，您可以在 `inference.py` 中更改 data_file ( test.jsonl ) 以及 output_path ( 用以儲存完整生成內容，包含 file_name、generated、label ) 的路徑。
3. 請使用以下指令做推理:
```bash
python inference.py checkpoint-4000_01
```
> 第一次推理，將自動下載 base model: `THUDM/glm-4v-9b`。推理 200 筆資料大約耗時 40 分鐘。

## 參考文獻
```
@misc{wang2023cogvlm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools}, 
      author={Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Jingyu Sun, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, Zihan Wang},
      year={2024},
      eprint={2406.12793},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
