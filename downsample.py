import os
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi
INPUT_SR = 56000  # Dataset asli 48kHz (bukan 56kHz)
TARGET_SR = 16000 # Target SERGAN
BASE_DIR = 'data'

# Daftar dataset yang perlu dikonversi
datasets = [
    'clean_trainset_56spk_wav',
    'noisy_trainset_56spk_wav', 
    'clean_testset_wav',
    'noisy_testset_wav'
]

def downsample_dataset(input_dir, output_dir):
    """Downsample semua file .wav di input_dir ke 16kHz"""
    if not os.path.exists(input_dir):
        print(f"❌ Folder tidak ditemukan: {input_dir}")
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    if not files:
        print(f"⚠️  Tidak ada file .wav di {input_dir}")
        return 0
    
    print(f"📁 Memproses {len(files)} file dari {input_dir}...")
    
    processed = 0
    for filename in tqdm(files, desc=os.path.basename(input_dir)):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Load audio dengan sampling rate asli
            y, sr = librosa.load(input_path, sr=INPUT_SR, mono=True)
            
            # Resample ke 16kHz
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            
            # Simpan dengan format 16-bit PCM (sama seperti SERGAN expect)
            sf.write(output_path, y_16k, TARGET_SR, subtype='PCM_16')
            processed += 1
            
        except Exception as e:
            print(f"\n❌ Error pada {filename}: {str(e)}")
    
    return processed

# Proses semua dataset
if __name__ == "__main__":
    print("🚀 Memulai downsampling ke 16kHz...")
    total_files = 0
    
    for dataset in datasets:
        input_dir = os.path.join(BASE_DIR, dataset)
        output_dir = os.path.join(BASE_DIR, f"{dataset}_16kHz")
        
        # Skip jika output sudah ada
        if os.path.exists(output_dir):
            print(f"⏭️  {output_dir} sudah ada, skip...")
            continue
            
        count = downsample_dataset(input_dir, output_dir)
        total_files += count
        print(f"✓ {dataset}: {count} file\n")
    
    print(f"✅ SELESAI! Total {total_files} file terkonversi.")
    
    # Buat file list untuk training/testing
    print("\n📝 Membuat file list...")
    with open(os.path.join(BASE_DIR, 'train_wav.txt'), 'w') as f:
        files = os.listdir(os.path.join(BASE_DIR, 'clean_trainset_56spk_wav_16kHz'))
        f.write('\n'.join(files))
    
    with open(os.path.join(BASE_DIR, 'test_wav.txt'), 'w') as f:
        files = os.listdir(os.path.join(BASE_DIR, 'clean_testset_wav_16kHz'))
        f.write('\n'.join(files))
    
    print("🎉 Semua persiapan dataset selesai!")

# import soundfile as sf
# y, sr = sf.read('data/clean_trainset_56spk_wav_16kHz/p234_001.wav')
# print(f"Sampling rate: {sr} Hz")  # Harusnya 16000
# print(f"Shape: {y.shape}")