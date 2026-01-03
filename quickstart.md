# 🚀 SERGAN PyTorch - Quick Start Guide

## 📁 File Structure

```
sergan_pytorch/
├── config.py          # 👈 EDIT KONFIGURASI DI SINI
├── run.py             # 👈 JALANKAN FILE INI
├── models.py          # (jangan diubah)
├── train.py           # (jangan diubah)
└── utils.py           # (jangan diubah)
```

## ⚡ Cara Menggunakan (3 Langkah)

### 1️⃣ Edit `config.py`

Buka `config.py` dan ubah opsi sesuai kebutuhan:

```python
# Di config.py:

MODE = 'train'  # 'train' atau 'test'
USE_DIRECTML = True  # True untuk AMD/Intel GPU
USE_SIMPLE_MODEL = True  # True untuk hemat RAM
BATCH_SIZE = 2  # 1-2 untuk laptop

# Path dataset
NOISY_TRAIN_DIR = './data/noisy_trainset_wav'
CLEAN_TRAIN_DIR = './data/clean_trainset_wav'
```

### 2️⃣ Run dengan CodeRunner

Tekan **Run** di VSCode atau jalankan:
```bash
python run.py
```

### 3️⃣ Done! 🎉

---

## 📖 Contoh Penggunaan

### 🔥 Scenario 1: Training dengan Dataset

**Edit `config.py`:**
```python
MODE = 'train'
USE_DIRECTML = True
USE_SIMPLE_MODEL = True
BASE_FILTERS = 8
BATCH_SIZE = 2
EPOCHS = 50

# Dataset paths
NOISY_TRAIN_DIR = './data/noisy_trainset_wav'
CLEAN_TRAIN_DIR = './data/clean_trainset_wav'
USE_SAMPLE_DATA = False
```

**Run:**
```bash
python run.py
```

---

### 🔥 Scenario 2: Testing dengan Sample Data (Coba-coba)

**Edit `config.py`:**
```python
MODE = 'train'
USE_DIRECTML = True
USE_SIMPLE_MODEL = True
BATCH_SIZE = 2
EPOCHS = 10

# Pakai sample data
USE_SAMPLE_DATA = True
NUM_SAMPLES = 50
```

**Run:**
```bash
python run.py
```

---

### 🔥 Scenario 3: Testing Single File

**Edit `config.py`:**
```python
MODE = 'test'
USE_DIRECTML = True
USE_SIMPLE_MODEL = True

TEST_MODE = 'single_file'
INPUT_FILE = 'noisy_audio.wav'
OUTPUT_FILE = 'enhanced_audio.wav'
CHECKPOINT_PATH = 'checkpoints/final_model.pt'
```

**Run:**
```bash
python run.py
```

---

### 🔥 Scenario 4: Testing Testset (Batch)

**Edit `config.py`:**
```python
MODE = 'test'
USE_DIRECTML = True
USE_SIMPLE_MODEL = True

TEST_MODE = 'testset'
TEST_NOISY_DIR = './data/noisy_testset_wav'
TEST_CLEAN_DIR = './data/clean_testset_wav'  # Optional
TEST_OUTPUT_DIR = './results/enhanced'
CHECKPOINT_PATH = 'checkpoints/final_model.pt'
```

**Run:**
```bash
python run.py
```

---

## ⚙️ Opsi Konfigurasi Penting

### Device Settings
```python
USE_DIRECTML = True   # True untuk AMD/Intel GPU
                      # False untuk CUDA (NVIDIA) atau CPU
```

### Model Settings
```python
USE_SIMPLE_MODEL = True   # True = simple (hemat RAM ~500MB)
                          # False = full (best quality ~4GB)

BASE_FILTERS = 8          # 4 = sangat hemat, 8 = balanced, 16 = best quality
```

### Training Settings
```python
GAN_TYPE = 'rasgan-gp'   # 'lsgan' = fastest
                         # 'wgan-gp' = stable
                         # 'rasgan-gp' = best quality (slow)

EPOCHS = 50              # 10-20 = quick test, 50-100 = good results
BATCH_SIZE = 2           # 1 = hemat RAM, 2-4 = balanced
```

### Memory Settings
```python
LAZY_LOAD = False        # False = load all to RAM (fast)
                         # True = load per batch (hemat RAM tapi lambat)

WINDOW_SIZE = 16384      # 8192 = hemat RAM, 16384 = standard
```

---

## 💾 Dataset Structure

Struktur folder yang diharapkan:

```
data/
├── noisy_trainset_wav/
│   ├── p226_001.wav
│   ├── p226_002.wav
│   └── ...
├── clean_trainset_wav/
│   ├── p226_001.wav
│   ├── p226_002.wav
│   └── ...
├── noisy_testset_wav/
│   └── ...
└── clean_testset_wav/
    └── ...
```

---

## 🎯 Rekomendasi untuk Laptop (Ryzen 5 5500U + iGPU)

### ✅ Setting Optimal:
```python
USE_DIRECTML = True
USE_SIMPLE_MODEL = True
BASE_FILTERS = 8
BATCH_SIZE = 2
EPOCHS = 50
LAZY_LOAD = False  # Jika RAM >=8GB
WINDOW_SIZE = 16384
```

### 🔋 Setting Hemat (jika lambat/crash):
```python
USE_DIRECTML = True
USE_SIMPLE_MODEL = True
BASE_FILTERS = 4
BATCH_SIZE = 1
EPOCHS = 30
LAZY_LOAD = True  # Hemat RAM
WINDOW_SIZE = 8192
```

---

## 🐛 Troubleshooting

### Out of Memory?
```python
# Di config.py:
BATCH_SIZE = 1
BASE_FILTERS = 4
LAZY_LOAD = True
WINDOW_SIZE = 8192
```

### Training terlalu lambat?
```python
# Di config.py:
GAN_TYPE = 'lsgan'  # Lebih cepat dari rasgan-gp
EPOCHS = 20
```

### DirectML error?
```bash
pip uninstall torch-directml
pip install torch-directml
```

Atau set:
```python
USE_DIRECTML = False  # Fallback ke CPU
```

---

## 📊 Monitoring Training

Saat training, Anda akan melihat:

```
Epoch 1/50: 100%|████████| 145/145 [02:15<00:00]
d_loss: 0.234  g_loss: 2.456  g_loss_adv: 0.123  l1_loss: 2.333

Epoch 1/50
D Loss: 0.2345
G Loss: 2.4567
G Adv Loss: 0.1234
L1 Loss: 2.3333
```

**Good signs:**
- D Loss stable (tidak naik/turun drastis)
- G Loss turun perlahan
- L1 Loss turun

---

## 🎓 Tips

1. **Mulai dengan sample data** untuk test setup
2. **Gunakan simple model** di laptop
3. **Start dengan epochs kecil** (10-20) untuk test
4. **Monitor GPU usage** di Task Manager
5. **Save checkpoints** otomatis setiap 10 epochs

---

## ✅ Checklist

- [ ] Install dependencies: `torch`, `librosa`, `soundfile`, `torch-directml`
- [ ] Download/prepare dataset
- [ ] Edit `config.py`
- [ ] Run `python run.py`
- [ ] Wait for training
- [ ] Test dengan testset
- [ ] Done! 🎉

---

**Selamat mencoba! 🚀**