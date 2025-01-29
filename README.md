# GLM-4v-8b-base-recruitment-advertisement

## 硬體設備
python: 3.10。
CUDA: 12.1。
GPU: 32GB以上。

## 環境設置
您可以使用以下指令來創建完整conda虛擬環境glm_py_3.10:
```bash
conda env create -f environment.yml
```

## 微調模型介紹
checkpoint-4000_01為使用label_0120中前823筆資料為訓練集微調，最後200筆為測試集，分數如下圖:
![metric]("metric.png")


## 模型推理
1.請先使用 `xlsx2jsonl.py` 將包含file_name和result的xlsx檔案轉換成text.jsonl。
+ 需更改input_file
2.若需要，您可以在 `inference.py` 中更改data_file(test.jsonl)以及output_path(用以儲存完整生成內容，包含file_name、generated、label)的路徑。

使用以下指令做推理:
```bash
python inference.py checkpoint-4000_01
```
> 第一次推理，將自動下載base model:THUDM/glm-4v-9b。