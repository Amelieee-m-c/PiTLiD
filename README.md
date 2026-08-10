# PiTLiD 重現 — 蘋果 4 類小樣本分類任務

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras_port-FF6F00?logo=tensorflow&logoColor=white)
![Reproduction](https://img.shields.io/badge/accuracy_gap-±0.7pp-2ea44f)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

重現的論文:
> "PiTLiD: Identification of Plant Disease From Leaf Images Based on Convolutional
> Neural Network", IEEE, 2022.(官方程式碼:https://github.com/zhanglab-wbgcas/PiTLiD,
> 本機副本在 `E:\plant_disease\PiTLiD`)

**2026-08-10 更新:改成採用官方程式碼。** 原本是 clean-room 從論文文字獨立
重新實作,理由是原作者的程式碼混雜 Keras/PyTorch、寫死路徑、用已棄用的
API,「無法直接執行」——但這不代表不能拿來讀,程式碼裡的實際邏輯遠比論文
文字可靠。`src/train_apple_pitlid.py` 現在已經照官方蘋果訓練腳本
(`differant setting/prediy7.py`)重寫,細節見下方「架構/協定修正」。

## 重現結果

蘋果 4 類分類,10 次獨立跑(`src/run_multi_seed.py`),平均值 ± 標準差。
新架構(照官方程式碼)跑了兩組學習率策略(見下方「LR 策略:CLR vs step
decay」),兩組都已完成:

| 指標 | 論文 | 舊版(論文文字猜的 CLR,單階段架構) | 新版 step-decay(官方 code) | 新版 CLR(官方 code + 真 CLR) |
|---|---|---|---|---|
| Accuracy | 99.45 ± 0.17% | 98.74 ± 0.47% | 98.67 ± 0.69% | 98.19 ± 1.00% |
| Precision | 98.84 ± 0.31% | 98.40 ± 0.92% | 98.08 ± 1.39% | 97.40 ± 1.62% |
| Sensitivity / Recall | 99.10 ± 0.23% | 98.42 ± 0.63% | 98.64 ± 0.77% | 98.22 ± 0.92% |
| F1 | 99.00 ± 0.23% | 98.38 ± 0.75% | 98.33 ± 1.10% | 97.77 ± 1.28% |

意外的結果:**CLR 版本反而是四組裡最差的**,直接跟論文 Figure 5B「CLR 是
三種 LR 策略裡最好的」這個主張相反。四組全部卡在 98.2%~98.7% 之間,沒有
一組真正逼近論文的 99.45%,而且標準差都明顯比論文的 0.17~0.31% 大——這
比較可能反映的是「30-shot 小樣本本身訓練不穩定」這個共通問題,而不是
LR 策略選錯。`clr_step_size`(CLR 半週期長度)因為官方 `clr_callback.py`
遺失、一直是用猜的(2000),很可能是 CLR 版本表現不如預期的原因——真正
決定 CLR 效果好壞的往往就是這個超參數,猜錯了 CLR 反而可能比簡單的
step decay 差,不代表 CLR 這個方法本身沒用。**目前沒有一個版本可以宣稱
「重現成功」**,四組都在同一個 ballpark,差距在雜訊範圍內。

另外也跑了論文的葡萄、桃子泛化資料集(單一 seed,Keras 版本,同樣是舊版):
葡萄 99.09% accuracy、桃子 99.54% accuracy,都很接近論文近乎完美的混淆矩陣。

## 架構/協定修正(改用官方程式碼後)

官方蘋果訓練腳本(`prediy7.py`)跟論文文字有好幾處對不上,已經逐一核對並
改寫:

| 項目 | 論文文字/我們原本猜的 | 官方程式碼實際寫的 |
|---|---|---|
| 分類頭 | GAP 直接接 Linear(關掉 dropout) | **GAP → Dropout(0.5) → ReLU → Dense(4, softmax)** |
| 輸入尺寸 | 299×299(Inception-V3 標準尺寸) | **256×256** |
| 訓練流程 | 單階段,一開始就全層微調 | **兩階段**:先凍結 backbone 只訓練分類頭(10 epoch, steps=100),再解凍全部層繼續訓練(40 epoch, steps=200) |
| Early stopping | patience=14,監控 val_loss,全程套用 | **實際上完全沒有作用**——`EarlyStopping` 物件在程式碼裡有定義,但從來沒被傳進 `fit_generator`,是死碼;兩階段都跑滿固定 epoch 數,只用 `ModelCheckpoint(monitor='val_accuracy')` 存最佳權重 |
| L2 正則化 | weight_decay 用猜的(1e-4) | 程式碼寫 `layer.W_regularizer = l2(1e-3)`,但這是 Keras 1.x 的舊寫法,Keras 2.x 讀的是 `kernel_regularizer`,對已建好的 layer 事後設定 `W_regularizer` 極可能是無效的死碼——預設改成 **weight_decay=0.0**,`--weight_decay` 可以另外測 1e-3 |
| 資料增強 | RandomAffine(±30°, shift 0.15, scale 0.85-1.15, shear 10°) | 更激進:**rotation 90°、shift 0.3、shear 0.3°、zoom 0.3(scale 0.7-1.3)**、水平+垂直翻轉 |

### LR 策略:CLR vs step decay(兩個都有官方程式碼證據,並列呈現)

這裡出現了跟 ConViTX 類似但更複雜的情況:論文 **Figure 5B 明確主張** CLR
(Cyclical LR,triangular2,base=0.001,max=0.006)是三種學習率策略裡最好的
(>99.31% accuracy,贏過固定 LR 跟 decay LR)。但目前 committed 到官方 repo
的 `prediy7.py`(蘋果專用腳本)裡,CLR 其實是**被註解掉的**,實際生效的是
簡單的兩段式 step decay(前 20 epoch lr=1e-3,之後 lr=1e-4)。

深入檢查後發現關鍵證據:

- `prediy7.py` 本身有 `from clr_callback import *`(匯入 CLR 的 callback
  模組),還有被註解掉的 `#clr_triangular = CyclicLR(mode='triangular')`
- 同一支腳本裡還留著兩個被註解掉的 **CLR 訓練出來的 checkpoint 檔名**:
  `weights_inceptionv3_peachclr_tanh.h5`(桃子)、
  `weights_inceptionv3_potatoclr1e-5_tanh.h5`(馬鈴薯)——證明 CLR 版本
  真的存在過、也真的訓練出其他作物的模型
- `clr_callback.py` 這個檔案本身**沒有被推上 GitHub**(git log 只有兩次
  commit,一開始 CLR 就已經是註解狀態),推測是作者用了本機/外部的檔案
  (很可能是常見的 bckenstler/CLR gist),忘記一起推上去

結論:CLR 不是論文文字說說而已,是真的被執行過、有明確產出證據的方法,
只是蘋果這支腳本目前 committed 的版本剛好停在 step decay 那個實驗狀態。
兩者都有真實的程式碼證據支持,不是「論文 vs 程式碼」的單純二選一,所以
`--lr_strategy` 開放兩種都能跑(預設 `clr`),兩組完整的 10-run 都會執行,
結果並列呈現,不指定哪一個是「正式」數字。`clr_step_size`(CLR 半週期
長度)因為 `clr_callback.py` 遺失,仍然是猜的(預設 2000)。

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
