
# 🧠 RFMiD CNN Classification — Retina Disease Detection with VGG & ResNet

本專案為暑期訓練 Week 2 的作業實作，使用 CNN 模型（VGG16、ResNet18、ResNet50）完成 RFMiD 眼底疾病資料集分類任務，並透過 loss 曲線與訓練準確率分析不同模型表現。使用 wandb 進行訓練監控，並導入 focal loss 解決類別不平衡問題。

---

## 🎯 學習目標

- 理解 VGG 與 ResNet 架構差異（參數量、訓練速度、效果）
- 熟悉圖像分類任務的訓練流程（loss function、optimizer、early stopping）
- 學習 focal loss 解決 class imbalance 問題
- 使用 [Weights & Biases (wandb)](https://wandb.ai/) 可視化訓練過程與超參數實驗結果

---

## 📁 專案結構

```
.
├── rfmid_dataset.py        # RFMiD 資料集定義
├── focal_loss.py           # Focal Loss 實作
├── train_cnn.py            # 主訓練程式
├── sweep.yaml              # wandb sweep 設定 (optional)
├── README.md
└── Retinal-disease-classification/
    ├── labels.csv          # 影像名稱與疾病標籤
    └── images/             # 影像檔案
```

---

## 📊 Dataset: RFMiD (Retinal Fundus Multi-Disease Image Dataset)

- 來源：[Kaggle](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification)
- 影像數量：3200 張眼底圖像
- 分類數：28 種視網膜疾病
- 資料格式：`.csv` 檔中包含圖片名稱與疾病類別對應

---

## 🧠 支援模型架構

| 模型名稱   | 參數數量 | 特點                         |
|------------|-----------|------------------------------|
| VGG16      | 138M      | 傳統大型卷積網路              |
| ResNet18   | 11M       | 較小但引入 residual block    |
| ResNet50   | 25M       | 更深更準確                   |

模型可自由切換訓練：

```python
model = get_model("resnet18")  # 可選：vgg16、resnet18、resnet50
```

---

## 🛠 使用方法

### 1️⃣ 安裝套件

```bash
pip install torch torchvision matplotlib pandas scikit-learn wandb
```

登入 wandb：

```bash
wandb login
```

---

### 2️⃣ 準備資料集

從 Kaggle 下載並放置在：

```
./Retinal-disease-classification/
├── labels.csv
└── images/
```

---

### 3️⃣ 執行訓練

```bash
python train_cnn.py
```

訓練過程將自動上傳到 wandb 項目中，可視化 loss 與 accuracy 曲線。

---

### 4️⃣ 使用 focal loss（optional）

```python
from focal_loss import FocalLoss
criterion = FocalLoss(alpha=1, gamma=2)
```

---

## 📈 訓練成果展示（wandb）

> 記得將 wandb 訓練連結附在這裡，例如：
- ResNet18 baseline: [wandb link]
- ResNet18 + FocalLoss: [wandb link]
- VGG16 baseline: [wandb link]

---

## 🧪 延伸任務建議

- 使用 data augmentation 提升泛化能力
- 使用 early stopping 與 scheduler 自動調整學習率
- 利用 `confusion matrix` 進行分類誤差分析
- 使用 wandb sweep 調整 learning rate / batch size 等超參數

---

## 🙌 貢獻者

- 👨‍🎓 訓練設計：你
- 📘 教學與程式撰寫：你
- 🎓 學生實作與報告：學生姓名

---

## 📜 License

本專案僅供教學與學術使用，資料與模型版權屬原作者所有。
