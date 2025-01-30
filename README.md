# GLM-4v-8b-base-recruitment-advertisement

## 設備需求
linux
python: 3.10。
CUDA: 12.1。
GPU: 32GB以上。

## 環境設置
您可以使用以下指令來創建完整 conda 虛擬環境`glm_py_3.10`，將創建到默認的`envs`目錄下:
```bash
conda env create -f environment.yml
```

## 微調模型介紹
`checkpoint-4000_01`為使用玉山提供之資料集`label_0120.xlsx`中前 823 筆資料為訓練集微調，最後 200 筆為測試集，分數如圖metric.png。


## 模型推理
1. 請先使用 `xlsx2jsonl.py` 將包含 file_name 和 result 的 xlsx 檔案轉換成`text.jsonl`。
+ 需更改 user_message 中的 "image"路徑
+ 需更改 input_file。
2. 若需要，您可以在 `inference.py` 中更改 data_file ( test.jsonl ) 以及 output_path ( 用以儲存完整生成內容，包含 file_name、generated、label ) 的路徑。
3. 請使用以下指令做推理:
```bash
python inference.py checkpoint-4000_01
```
> 第一次推理，將自動下載 base model: `THUDM/glm-4v-9b`。推理大約耗時30分鐘。