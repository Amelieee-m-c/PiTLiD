# PiTLiD 重現 — 蘋果 4 類小樣本分類任務

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras_port-FF6F00?logo=tensorflow&logoColor=white)
![Reproduction](https://img.shields.io/badge/accuracy_gap-±0.7pp-2ea44f)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

重現的論文:
> "PiTLiD: Identification of Plant Disease From Leaf Images Based on Convolutional
> Neural Network", IEEE, 2022.(官方程式碼:https://github.com/zhanglab-wbgcas/PiTLiD)

這是一份獨立的 clean-room PyTorch 重新實作,與原作者的 repo 無關——原作者的
程式碼混雜 Keras/PyTorch、寫死路徑、使用已棄用的 API,無法直接執行。

## 重現結果

蘋果 4 類分類,10 次獨立跑(`src/run_multi_seed.py`),平均值 ± 標準差:

| 指標 | 論文 | 重現結果 |
|---|---|---|
| Accuracy | 99.45 ± 0.17% | 98.74 ± 0.47% |
| Precision | 98.84 ± 0.31% | 98.40 ± 0.92% |
| Sensitivity / Recall | 99.10 ± 0.23% | 98.42 ± 0.63% |
| F1 | 99.00 ± 0.23% | 98.38 ± 0.75% |

另外也跑了論文的葡萄、桃子泛化資料集(單一 seed,Keras 版本):葡萄
99.09% accuracy、桃子 99.54% accuracy,都很接近論文近乎完美的混淆矩陣。

差距推測來自論文沒有明確給出數值的兩個超參數(CLR 半週期長度、L2
weight-decay 強度——都是用猜的,詳見下方「已知偏差」段落)。

專案裡也附上一份平行的 Keras/TensorFlow 版本(`src/train_apple_pitlid_keras.py`),
用來做框架一致性的對照,但在這台機器上是 CPU-only 執行——TensorFlow ≥2.11
已經不支援原生 Windows 的 GPU 加速。時間敏感的場合請用 PyTorch 版本。

## 資料處理流程

1. **來源**:`E:\plant_disease\rethinking_fewshot_vlms\data\PlantVillage_Split_721`
   ——完整 38 類、54,306 張圖片的 PlantVillage "color" 資料集,已經切好的 7:2:1
   train/val/test。**只讀取,沒有修改**。
2. **合併**(`robocopy`,一次性操作):把三個 split 的圖片複製回
   `E:\plant_disease\PlantVillage_full\<class>\*`,重建論文取樣協定假設的
   未切分、池化資料集。
3. **論文的小樣本切分** — `data_prep/make_pitlid_apple_split.py`:針對蘋果的
   4 個類別,各抽 30 張圖當 `train`(有設定隨機種子),剩下的圖片約 1:1 分成
   `val`/`test`(數量為奇數時多的那張分給 val,對應論文 Table 1)。輸出到
   `data/apple_pitlid_split/{train,val,test}/<class>/*`。

   已對照論文 Table 1 驗證過(seed=1):

   | 類別 | train | val | test | 總計 |
   |---|---|---|---|---|
   | Apple_scab | 30 | 300 | 300 | 630 |
   | Black_rot | 30 | 296 | 295 | 621 |
   | Cedar_apple_rust | 30 | 123 | 122 | 275 |
   | healthy | 30 | 808 | 807 | 1645 |

## 模型與訓練(`src/train_apple_pitlid.py`)

- Backbone:`torchvision` 的 Inception-V3,ImageNet 預訓練,**全部層都微調**
  (對應論文表現最好的 "None_frozen" 策略)。
- 分類頭:GAP 直接接 `Linear(2048, 4)` + softmax(關掉 dropout,對應論文說的
  "GAP connects directly to the output layer";關掉 aux classifier,因為 Keras
  的 `InceptionV3(include_top=False)` 本來就沒有這個)。
- 前處理:resize 到 299×299,`ToTensor()`(rescale 到 [0,1],即除以 255)——
  沒有做 ImageNet mean/std normalize,照論文規格。如果收斂有問題,可以加
  `--imagenet_norm` 把 normalize 加回去。
- 資料增強(僅訓練集):隨機水平/垂直翻轉 + 一個結合 rotation/shift/zoom/shear
  的 `RandomAffine`。
- Optimizer:RMSprop,`weight_decay` 對應 L2 正則化。
- Batch size:train 32 / val 16。Epochs 50,steps/epoch 200(訓練集每個
  epoch 都用取後放回的方式重新取樣,對應 Keras 在 120 張圖的目錄上用
  `ImageDataGenerator` + `steps_per_epoch` 的行為)。
- 學習率:Cyclical LR,`mode="triangular2"`,`base_lr=0.001`,
  `max_lr=0.006`(`--clr_step_size`,預設每半週期 2000 個 batch——論文摘錄
  裡沒有明確給這個數字,如果你知道確切數值可以自行調整)。
- Early stopping:監控 `val_loss`,patience 14,還原最佳權重。
- 測試集指標:accuracy、macro precision/recall(=sensitivity)/F1、混淆矩陣
  (原始值+標準化,存成 PNG)、完整的 `sklearn.classification_report`。

## 10-run 穩定性協定(`src/run_multi_seed.py`)

論文報告的結果是 **10 次獨立跑** 的平均(蘋果的平均準確率 99.45%)。這裡的每一
次跑都會重新抽一次 30-shot 切分**並且**用該次的 seed 從頭訓練,再把 10 次的
測試集指標算成 mean ± std。這是對 30-shot 協定下「10 次獨立跑」比較保守的
解讀方式(同時涵蓋了「抽到哪 30 張圖」的變異,而不只是訓練 seed 的變異)——
如果論文的意思是固定同一組切分重複跑 10 次,可以調整 `run_multi_seed.py`。

```
python data_prep/make_pitlid_apple_split.py --seed 1
python src/train_apple_pitlid.py --data_dir data/apple_pitlid_split --output_dir runs/apple_seed1

# 完整論文協定,GPU 上約 4-6 小時:
python src/run_multi_seed.py --n_runs 10 --output_root runs/apple_10run
```

## 已知偏差 / 需要對照論文 PDF 再次確認的假設

- `clr_step_size`(CLR 半週期的 batch 數):預設 2000,目前拿到的論文摘錄
  沒有給出這個數字。
- L2 weight_decay 強度:預設 `1e-4`,論文沒有明確給出。
- Early stopping 監控的是 `val_loss`(Keras 預設行為);要再確認論文是否
  改監控 training loss。
- 10-run 協定每次都重新抽 30-shot 切分(見上方說明)——要再確認論文是否
  其實是 10 次都用同一組切分。
- 葡萄/桃子泛化資料集(同樣是 30-shot):已經實作並跑過(單一 seed,
  Keras 版本)——結果見上方。

## 環境需求

Python 3.10+、PyTorch + torchvision(建議 CUDA 版)、scikit-learn、
matplotlib。Keras 版本額外需要 TensorFlow(在原生 Windows 上只能跑
CPU——TF ≥2.11 已經不支援原生 Windows 的 GPU 加速,要用 GPU 得裝 WSL2)。

## 授權

本 repo 程式碼採用 MIT 授權。不包含任何資料集圖片——資料來源請見上方
「資料處理流程」,各自有其原始授權。
