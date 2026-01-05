"""
SERGAN PyTorch - Main Runner
Jalankan file ini dengan CodeRunner atau: python run.py

Edit konfigurasi di config.py
"""

from pathlib import Path
from config import Config

# Import modules
# from models import Generator, Discriminator, SimpleGenerator, SimpleDiscriminator
from models_v2 import Generator, Discriminator
# from train import train_sergan
from train_v2 import train_sergan
from utils import (
    get_device, load_audio, save_audio, enhance_audio,
    prepare_dataset, create_sample_data, ModelSizeCalculator,
    evaluate_testset
)


def train(resume_from=None):
    """Training function"""
    print("="*70)
    print("SERGAN TRAINING")
    print("="*70)
    
    # Print configuration
    Config.print_config()
    
    # Get device
    device = get_device(prefer_directml=Config.USE_DIRECTML)
    
    # Prepare data
    if Config.USE_SAMPLE_DATA:
        print("\n📊 Creating sample data for testing...")
        noisy_data, clean_data = create_sample_data(
            num_samples=Config.NUM_SAMPLES,
            sample_length=Config.WINDOW_SIZE,
            sr=Config.SAMPLE_RATE,
            apply_preemph=Config.APPLY_PREEMPH,
            preemph_coeff=Config.PREEMPH_COEFF
        )
        lazy_load = False
        print(f"✅ Sample dataset created: {noisy_data.shape}")
    else:
        print("\n📊 Loading real dataset...")
        data = prepare_dataset(
            Config.NOISY_TRAIN_DIR, 
            Config.CLEAN_TRAIN_DIR, 
            sr=Config.SAMPLE_RATE, 
            window_size=Config.WINDOW_SIZE,
            lazy_load=Config.LAZY_LOAD,
            apply_preemph=Config.APPLY_PREEMPH,
            preemph_coeff=Config.PREEMPH_COEFF
        )
        
        if Config.LAZY_LOAD:
            noisy_data, clean_data = data
            lazy_load = True
            print("✅ Using lazy loading (will load from disk per batch)")
        else:
            noisy_data, clean_data = data
            lazy_load = False
            print(f"✅ Dataset loaded to memory. Shape: {noisy_data.shape}")
    
    # Create models
    print("\n🔧 Initializing models...")
    # if Config.USE_SIMPLE_MODEL:
    #     print("   Using SIMPLE models (less memory)")
    #     generator = SimpleGenerator(
    #         input_channels=1, 
    #         output_channels=1, 
    #         base_filters=Config.BASE_FILTERS
    #     )
    #     discriminator = SimpleDiscriminator(
    #         input_channels=2, 
    #         base_filters=Config.BASE_FILTERS
    #     )
    # else:
    print("   Using FULL models (best quality)")
    generator = Generator(
        input_channels=1, 
        output_channels=1, 
        base_filters=Config.BASE_FILTERS
    )
    discriminator = Discriminator(
        input_channels=2, 
        base_filters=Config.BASE_FILTERS
    )
    
    # Print model info
    ModelSizeCalculator.print_model_info(generator, "Generator")
    ModelSizeCalculator.print_model_info(discriminator, "Discriminator")
    
    # Train
    print(f"🚀 Starting training...")
    if resume_from:
        print(f"   Resume from checkpoint: {resume_from}")
    print(f"   GAN Type: {Config.GAN_TYPE}")
    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Lazy Loading: {lazy_load}")
    print()
    
    trainer = train_sergan(
        train_noisy=noisy_data,
        train_clean=clean_data,
        generator=generator,
        discriminator=discriminator,
        device=device,
        gan_type=Config.GAN_TYPE,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        save_dir=Config.SAVE_DIR,
        lazy_load=lazy_load,
        apply_preemph=Config.APPLY_PREEMPH,
        preemph_coeff=Config.PREEMPH_COEFF,
        use_spec_loss=Config.USE_SPEC_LOSS,
        spec_loss_weight=Config.SPEC_LOSS_WEIGHT,
        checkpoint_path=resume_from,
        use_envelope_loss=Config.USE_ENVELOPE_LOSS,
        envelope_loss_weight=Config.ENVELOPE_LOSS_WEIGHT
    )
    
    # Save final model
    final_path = Path(Config.SAVE_DIR) / 'final_model.pt'
    trainer.save_checkpoint(final_path, Config.EPOCHS, {})
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print(f"📁 Final model saved to: {final_path}")
    print("="*70 + "\n")


def test():
    """Testing/Enhancement function"""
    print("="*70)
    print("SERGAN TESTING")
    print("="*70)
    
    # Print configuration
    Config.print_config()
    
    # Get device
    device = get_device(prefer_directml=Config.USE_DIRECTML)

    # 1. HARUS BUAT OBJEK GENERATOR DULU (Sesuai config)
    # if Config.USE_SIMPLE_MODEL:
    #     generator = SimpleGenerator(base_filters=Config.BASE_FILTERS).to(device)
    # else:
    generator = Generator(base_filters=Config.BASE_FILTERS).to(device)
    
    # Load model
    print(f"📂 Loading model from {Config.CHECKPOINT_PATH}...")
    try:
        checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at {Config.CHECKPOINT_PATH}")
        print("   Please train a model first or check the path!")
        return
    
    # Create generator
    # if Config.USE_SIMPLE_MODEL:
    #     generator = SimpleGenerator(
    #         input_channels=1, 
    #         output_channels=1,
    #         base_filters=Config.BASE_FILTERS
    #     )
    # else:
    generator = Generator(
        input_channels=1, 
        output_channels=1,
        base_filters=Config.BASE_FILTERS
    )
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator = generator.to(device)
    generator.eval()
    
    print("✅ Model loaded successfully!")
    ModelSizeCalculator.print_model_info(generator, "Generator")
    
    # Test based on mode
    if Config.TEST_MODE == 'single_file':
        # Single file enhancement
        print(f"\n🎵 Loading noisy audio from {Config.INPUT_FILE}...")
        try:
            noisy_audio = load_audio(Config.INPUT_FILE, sr=Config.SAMPLE_RATE,
                                        apply_preemph=Config.APPLY_PREEMPH,
                                        preemph_coeff=Config.PREEMPH_COEFF)
            print(f"   Duration: {len(noisy_audio)/Config.SAMPLE_RATE:.2f} seconds")
            
            # Enhance
            print("\n⚙️  Enhancing audio...")
            enhanced_audio = enhance_audio(
                model=generator,
                noisy_audio=noisy_audio,
                device=device,
                window_size=Config.WINDOW_SIZE,
                overlap=Config.OVERLAP
            )
            
            # Save
            print(f"💾 Saving enhanced audio to {Config.OUTPUT_FILE}...")
            save_audio(enhanced_audio, Config.OUTPUT_FILE, sr=Config.SAMPLE_RATE,
                            apply_deemph=Config.APPLY_PREEMPH,
                            preemph_coeff=Config.PREEMPH_COEFF)
                
            print("\n" + "="*70)
            print("✅ ENHANCEMENT COMPLETED!")
            print(f"📁 Output saved to: {Config.OUTPUT_FILE}")
            print("="*70 + "\n")
            
        except FileNotFoundError:
            print(f"❌ Error: Input file not found at {Config.INPUT_FILE}")
            return
    
    elif Config.TEST_MODE == 'testset':
        # Testset evaluation
        print(f"\n📊 Evaluating on testset...")
        print(f"   Noisy dir: {Config.TEST_NOISY_DIR}")
        if Config.TEST_CLEAN_DIR:
            print(f"   Clean dir: {Config.TEST_CLEAN_DIR}")
        print(f"   Output dir: {Config.TEST_OUTPUT_DIR}")
        print()
        
        results = evaluate_testset(
            generator=generator,
            noisy_dir=Config.TEST_NOISY_DIR,
            clean_dir=Config.TEST_CLEAN_DIR,
            output_dir=Config.TEST_OUTPUT_DIR,
            device=device,
            sr=Config.SAMPLE_RATE,
            window_size=Config.WINDOW_SIZE,
            overlap=Config.OVERLAP,
            apply_preemph=Config.APPLY_PREEMPH,
            preemph_coeff=Config.PREEMPH_COEFF
        )
        
        print("\n" + "="*70)
        print("✅ TESTSET EVALUATION COMPLETED!")
        print("="*70)
        print(f"📊 Total files processed: {results['total_files']}")
        if results['avg_snr_improvement'] is not None:
            print(f"📈 Average SNR improvement: {results['avg_snr_improvement']:.2f} dB")
        print(f"📁 Output saved to: {Config.TEST_OUTPUT_DIR}")
        print("="*70 + "\n")


if __name__ == '__main__':
    import torch
    
    if Config.MODE == 'train':
        train(resume_from=Config.RESUME_FROM)
    elif Config.MODE == 'test':
        test()
    else:
        print(f"❌ Error: Unknown mode '{Config.MODE}'")
        print("   Please set MODE to 'train' or 'test' in config.py")