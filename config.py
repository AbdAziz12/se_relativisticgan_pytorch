"""
Configuration file untuk SERGAN
Edit opsi-opsi di sini, lalu jalankan run.py
"""

class Config:
    """Configuration class untuk SERGAN"""
    
    # =================== MODE OPERASI ===================
    # Pilih mode: 'train' atau 'test'
    MODE = 'test'  # 👈 UBAH DI SINI: 'train' atau 'test'
    
    # =================== DEVICE SETTINGS ===================
    USE_DIRECTML = False  # 👈 True untuk AMD/Intel GPU, False untuk CUDA/CPU
    
    # =================== MODEL SETTINGS ===================
    # True = model simple (hemat RAM), False = model full (best quality)
    USE_SIMPLE_MODEL = False  # 👈 UBAH DI SINI
    BASE_FILTERS = 16  # 👈 16 untuk full quality, 8 atau 4 untuk hemat memory
    
    # =================== TRAINING SETTINGS ===================
    # GAN Type: 'lsgan', 'wgan-gp', 'rsgan-gp', 'rasgan-gp', 'ralsgan-gp'
    GAN_TYPE = 'rasgan-gp'  # 👈 UBAH DI SINI
    
    EPOCHS = 81  # 👈 Jumlah epoch training (asli 81)
    BATCH_SIZE = 64  # 👈 (asli 100)
    
    USE_SPEC_LOSS = True
    SPEC_LOSS_WEIGHT = 0.2

    USE_ENVELOPE_LOSS = False  # 👈 Aktifkan envelope loss
    ENVELOPE_LOSS_WEIGHT = 0.05  # 👈 Start dengan 0.04 (2% dari L1 weight 200)

    # Resume Training
    RESUME_FROM = None

    # =================== DATASET SETTINGS ===================
    # Path ke dataset Anda
    NOISY_TRAIN_DIR = './data/noisy_trainset_56spk_wav_16kHz'  # 👈 Path dataset noisy train
    CLEAN_TRAIN_DIR = './data/clean_trainset_56spk_wav_16kHz'  # 👈 Path dataset clean train
    
    # Lazy load: True = load dari disk per batch (hemat RAM), False = load semua ke memory
    LAZY_LOAD = False  # 👈 UBAH DI SINI jika RAM terbatas
    
    # Sample data (jika tidak punya dataset)
    USE_SAMPLE_DATA = False  # 👈 True = pakai sample data untuk testing
    NUM_SAMPLES = 50  # Jumlah sample data jika USE_SAMPLE_DATA = True
    
    # =================== AUDIO SETTINGS ===================
    SAMPLE_RATE = 16000
    WINDOW_SIZE = 16384  # 8192 untuk hemat memory, 16384 untuk quality
    OVERLAP = 0.5  # Overlap ratio untuk enhancement

    APPLY_PREEMPH = False
    PREEMPH_COEFF = 0.95
    
    # =================== SAVE SETTINGS ===================
    SAVE_DIR = 'checkpoints'  # Directory untuk save checkpoints
    
    # =================== TESTING SETTINGS ===================
    # Pilih mode testing: 'single_file' atau 'testset'
    TEST_MODE = 'single_file'
    # TEST_MODE = 'testset'
    
    # Single file mode
    INPUT_FILE = './my_speech_pink.wav'  # 👈 Input file untuk enhancement
    OUTPUT_FILE = './enhanced_single_file/model_rasgan_10_v2_pink.wav'  # 👈 Output file
    
    # Testset mode
    TEST_NOISY_DIR = './data/noisy_testset_wav_16kHz'  # 👈 Path testset noisy
    TEST_CLEAN_DIR = './data/clean_testset_wav_16kHz'  # 👈 Path testset clean (optional, untuk SNR)
    TEST_OUTPUT_DIR = './results/model_v2/model_rasgan_10_v2'  # 👈 Output directory
    
    # Model checkpoint untuk testing
    CHECKPOINT_PATH = 'checkpoints_exp/model_rasgan_10_v2.pt'  # 👈 Path ke trained model
    
    # =================== ADVANCED SETTINGS ===================
    SAVE_EVERY_N_EPOCHS = 5  # Save checkpoint setiap N epochs (asli 10)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*70)
        print("SERGAN Configuration")
        print("="*70)
        print(f"Mode: {cls.MODE}")
        print(f"Device: {'DirectML' if cls.USE_DIRECTML else 'CUDA/CPU'}")
        print(f"Model: {'Simple' if cls.USE_SIMPLE_MODEL else 'Full'}")
        print(f"Base Filters: {cls.BASE_FILTERS}")
        print(f"Preemphasis: {'Enabled' if cls.APPLY_PREEMPH else 'Disabled'} (coeff={cls.PREEMPH_COEFF})")
        
        if cls.MODE == 'train':
            print(f"\nTraining Settings:")
            print(f"  GAN Type: {cls.GAN_TYPE}")
            print(f"  Epochs: {cls.EPOCHS}")
            print(f"  Batch Size: {cls.BATCH_SIZE}")
            print(f"  Lazy Load: {cls.LAZY_LOAD}")
            print(f"  Spectral Loss: {'Enabled' if cls.USE_SPEC_LOSS else 'Disabled'}")
            if cls.USE_SPEC_LOSS:
                print(f"    - Weight: {cls.SPEC_LOSS_WEIGHT}")
            if cls.USE_SAMPLE_DATA:
                print(f"  Dataset: Sample data ({cls.NUM_SAMPLES} samples)")
            else:
                print(f"  Noisy Dir: {cls.NOISY_TRAIN_DIR}")
                print(f"  Clean Dir: {cls.CLEAN_TRAIN_DIR}")
        
        elif cls.MODE == 'test':
            print(f"\nTesting Settings:")
            print(f"  Test Mode: {cls.TEST_MODE}")
            if cls.TEST_MODE == 'single_file':
                print(f"  Input: {cls.INPUT_FILE}")
                print(f"  Output: {cls.OUTPUT_FILE}")
            else:
                print(f"  Noisy Dir: {cls.TEST_NOISY_DIR}")
                print(f"  Clean Dir: {cls.TEST_CLEAN_DIR}")
                print(f"  Output Dir: {cls.TEST_OUTPUT_DIR}")
            print(f"  Checkpoint: {cls.CHECKPOINT_PATH}")
        
        print("="*70 + "\n")