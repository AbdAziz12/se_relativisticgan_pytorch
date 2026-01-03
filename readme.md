# SERGAN PyTorch Implementation

Implementasi PyTorch dari **SERGAN (Speech Enhancement using Relativistic GANs)** yang kompatibel dengan CUDA, DirectML (AMD/Intel GPU), dan CPU.

## 🎯 **CARA CEPAT: Edit config.py → Run run.py!**

**Tidak perlu command line panjang!** Cukup:
1. Edit opsi di `config.py`
2. Run `python run.py` atau tekan Run di VSCode
3. Done! 🎉

👉 **Lihat [QUICKSTART.md](QUICKSTART.md) untuk panduan lengkap**

---

## 📋 Features

- ✅ Implementasi lengkap dari paper SERGAN (ICASSP 2019)
- ✅ Support untuk berbagai GAN variants:
  - LSGAN (Least Squares GAN)
  - WGAN-GP (Wasserstein GAN with Gradient Penalty)
  - RSGAN-GP (Relativistic Standard GAN with GP)
  - RaSGAN-GP (Relativistic average Standard GAN with GP)
  - RaLSGAN-GP (Relativistic average Least Squares GAN with GP)
- ✅ Kompatibel dengan CUDA, DirectML, dan CPU
- ✅ Model sederhana untuk laptop dengan resource terbatas
- ✅ Automatic device detection

## 🔧 Requirements

```bash
# Core requirements
pip install torch torchvision torchaudio
pip install numpy scipy librosa soundfile tqdm

# Untuk DirectML (AMD/Intel GPU di Windows)
pip install torch-directml

# Optional: untuk dataset processing
pip install matplotlib
```

## 📁 File Structure

```
sergan_pytorch/
├── config.py           # 👈 EDIT KONFIGURASI DI SINI
├── run.py              # 👈 JALANKAN FILE INI (python run.py)
├── models.py           # Generator & Discriminator
├── train.py            # Training logic
├── utils.py            # Utility functions
├── main.py             # CLI version (optional, untuk command line)
├── README.md           # Documentation
├── QUICKSTART.md       # Quick start guide
└── requirements.txt    # Dependencies
```

## 🚀 Quick Start

### Method 1: Edit Config & Run (RECOMMENDED ⭐)

```bash
# 1. Edit config.py
# Ubah opsi seperti MODE, dataset paths, dll

# 2. Run
python run.py

# That's it! ✅
```

### Method 2: Command Line (Advanced)

```bash
# Training
python main.py --mode train --noisy-dir ./data/noisy_trainset_wav --clean-dir ./data/clean_trainset_wav --simple --use-directml

# Testing
python main.py --mode test --test-noisy-dir ./data/noisy_testset_wav --checkpoint checkpoints/final_model.pt --use-directml
```

---

### 1. Training dengan Sample Data (untuk testing)

```bash
# CPU
python main.py --mode train --simple --epochs 10 --batch-size 2

# DirectML (untuk AMD/Intel GPU)
python main.py --mode train --simple --use-directml --epochs 10 --batch-size 2

# CUDA (untuk NVIDIA GPU)
python main.py --mode train --simple --epochs 10 --batch-size 4
```

### 2. Training dengan Real Dataset

```bash
# Download dataset Valentini et al. terlebih dahulu
# Kemudian:

python main.py --mode train \
    --noisy-dir /path/to/noisy_trainset_wav \
    --clean-dir /path/to/clean_trainset_wav \
    --gan-type rasgan-gp \
    --epochs 100 \
    --batch-size 4 \
    --use-directml
```

### 3. Enhancement/Testing

**A. Single File Enhancement**
```bash
python main.py --mode test \
    --input noisy_audio.wav \
    --output enhanced_audio.wav \
    --checkpoint checkpoints/final_model.pt \
    --use-directml
```

**B. Testset Evaluation (Batch Processing)**
```bash
# Without clean reference (just enhance)
python main.py --mode test \
    --test-noisy-dir /path/to/noisy_testset_wav \
    --test-output-dir enhanced_testset \
    --checkpoint checkpoints/final_model.pt \
    --use-directml

# With clean reference (calculate SNR improvement)
python main.py --mode test \
    --test-noisy-dir /path/to/noisy_testset_wav \
    --test-clean-dir /path/to/clean_testset_wav \
    --test-output-dir enhanced_testset \
    --checkpoint checkpoints/final_model.pt \
    --use-directml
```

## ⚙️ Command Line Arguments

### Mode
- `--mode`: Mode operasi (`train` atau `test`)

### Device
- `--use-directml`: Gunakan DirectML untuk AMD/Intel GPU

### Model
- `--simple`: Gunakan model sederhana (hemat memory)
- `--base-filters`: Jumlah base filters (default: 16)

### Training
- `--gan-type`: Tipe GAN (`lsgan`, `wgan-gp`, `rsgan-gp`, `rasgan-gp`, `ralsgan-gp`)
- `--epochs`: Jumlah epoch training (default: 100)
- `--batch-size`: Batch size (default: 4)
- `--save-dir`: Directory untuk save checkpoints (default: `checkpoints`)

### Data
- `--noisy-dir`: Directory dengan noisy audio files
- `--clean-dir`: Directory dengan clean audio files
- `--num-samples`: Jumlah sample data jika tidak ada real data (default: 100)

### Audio
- `--sample-rate`: Sample rate (default: 16000)
- `--window-size`: Window size untuk processing (default: 16384)
- `--overlap`: Overlap ratio untuk enhancement (default: 0.5)

### Testing
- `--input`: Input noisy audio file (untuk single file)
- `--output`: Output enhanced audio file (default: `enhanced.wav`)
- `--checkpoint`: Checkpoint file untuk load model
- `--test-noisy-dir`: Directory dengan test noisy files (untuk testset evaluation)
- `--test-clean-dir`: Directory dengan test clean files (optional, untuk SNR calculation)
- `--test-output-dir`: Output directory untuk enhanced test files (default: `enhanced_testset`)

## 💻 Spesifikasi untuk Laptop Anda (Ryzen 5 5500U)

### Recommended Settings

```bash
# Training dengan model simple
python main.py --mode train \
    --simple \
    --use-directml \
    --gan-type rasgan-gp \
    --epochs 50 \
    --batch-size 2 \
    --base-filters 8 \
    --num-samples 50

# Jika masih terlalu berat, reduce lebih lanjut:
python main.py --mode train \
    --simple \
    --use-directml \
    --gan-type lsgan \
    --epochs 20 \
    --batch-size 1 \
    --base-filters 8
```

### Tips untuk Optimasi
1. Gunakan `--simple` flag untuk model yang lebih kecil
2. Reduce `--base-filters` dari 16 ke 8 atau 4
3. Gunakan `--batch-size 1` atau `2`
4. Pilih `lsgan` atau `wgan-gp` untuk training lebih cepat
5. Set `--window-size 8192` untuk reduce memory usage

## 📊 Model Comparison

| Model | Parameters | Memory | Speed | Quality |
|-------|-----------|--------|-------|---------|
| Full Generator | ~8M | ~4GB | Slow | Best |
| Simple Generator | ~2M | ~1GB | Medium | Good |
| Simple + reduce filters | ~500K | ~500MB | Fast | Acceptable |

## 🎯 GAN Type Comparison

| GAN Type | Training Speed | Stability | Quality |
|----------|---------------|-----------|---------|
| LSGAN | Fast | Good | Good |
| WGAN-GP | Medium | Very Good | Good |
| RSGAN-GP | Medium | Good | Better |
| RaSGAN-GP | Slow | Very Good | Best |
| RaLSGAN-GP | Slow | Very Good | Best |

## 📝 Examples

### Example 1: Complete Workflow dengan Dataset Valentini

```bash
# 1. Training dengan trainset
python main.py --mode train \
    --noisy-dir ./data/noisy_trainset_wav \
    --clean-dir ./data/clean_trainset_wav \
    --simple --use-directml \
    --batch-size 2 --epochs 50 \
    --save-dir checkpoints

# 2. Evaluate pada testset
python main.py --mode test \
    --test-noisy-dir ./data/noisy_testset_wav \
    --test-clean-dir ./data/clean_testset_wav \
    --test-output-dir results/enhanced \
    --checkpoint checkpoints/final_model.pt \
    --use-directml

# 3. Enhance single file
python main.py --mode test \
    --input my_noisy_audio.wav \
    --output my_clean_audio.wav \
    --checkpoint checkpoints/final_model.pt \
    --use-directml
```

### Example 2: Quick Test dengan Sample Data

```python
# test_sample.py
import torch
from models import SimpleGenerator
from utils import create_sample_data, enhance_audio, save_audio, get_device

# Create sample data
noisy_data, clean_data = create_sample_data(num_samples=1)

# Get device
device = get_device(prefer_directml=True)

# Create and load model (setelah training)
generator = SimpleGenerator().to(device)
checkpoint = torch.load('checkpoints/final_model.pt', map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])

# Enhance
enhanced = enhance_audio(generator, noisy_data[0], device)
save_audio(enhanced, 'enhanced_sample.wav')
```

### Example 2: Batch Enhancement

```python
# batch_enhance.py
from pathlib import Path
from utils import load_audio, enhance_audio, save_audio
from models import SimpleGenerator
import torch

device = torch.device('cuda')  # atau DirectML
generator = SimpleGenerator().to(device)
checkpoint = torch.load('model.pt', map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])

input_dir = Path('noisy_files')
output_dir = Path('enhanced_files')
output_dir.mkdir(exist_ok=True)

for audio_file in input_dir.glob('*.wav'):
    print(f"Processing {audio_file.name}...")
    noisy = load_audio(str(audio_file))
    enhanced = enhance_audio(generator, noisy, device)
    save_audio(enhanced, output_dir / audio_file.name)
```

## 🐛 Troubleshooting

### DirectML Issues
```bash
# Jika error dengan DirectML, coba reinstall:
pip uninstall torch-directml
pip install torch-directml
```

### Out of Memory
```bash
# Reduce batch size dan model size:
python main.py --mode train --simple --batch-size 1 --base-filters 4
```

### Slow Training
```bash
# Gunakan LSGAN yang lebih cepat:
python main.py --mode train --simple --gan-type lsgan --batch-size 2
```

## 📚 References

Original paper:
```
Deepak Baby and Sarah Verhulst, 
"SERGAN: Speech enhancement using relativistic generative adversarial networks with gradient penalty", 
IEEE-ICASSP, pp. 106-110, May 2019, Brighton, UK.
```

Original Keras Implementation: https://github.com/deepakbaby/se_relativisticgan

## 📄 License

MIT License - feel free to use for your projects!

## 🤝 Contributing

Contributions welcome! Feel free to open issues or submit PRs.

## ⭐ Acknowledgments

- Original SERGAN implementation by Deepak Baby
- PyTorch team for excellent framework
- DirectML team for AMD/Intel GPU support