"""
SERGAN PyTorch Implementation
Speech Enhancement using Relativistic GANs

Compatible dengan:
- CUDA (NVIDIA GPU)
- DirectML (AMD/Intel GPU)  
- CPU

Usage:
    # Training dengan sample data
    python main.py --mode train --use-directml
    
    # Training dengan real data
    python main.py --mode train --noisy-dir /path/to/noisy --clean-dir /path/to/clean
    
    # Testing/Enhancement
    python main.py --mode test --input audio.wav --output enhanced.wav --checkpoint checkpoint.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path

# Import modules kita
from models import Generator, Discriminator, SimpleGenerator, SimpleDiscriminator
from train import train_sergan
from utils import (
    get_device, load_audio, save_audio, enhance_audio,
    prepare_dataset, create_sample_data, ModelSizeCalculator
)


def train(args):
    """Training function"""
    print("="*60)
    print("SERGAN Training")
    print("="*60)
    
    # Get device
    device = get_device(prefer_directml=args.use_directml)
    
    # Prepare data
    if args.noisy_dir and args.clean_dir:
        print("\nLoading real dataset...")
        data = prepare_dataset(
            args.noisy_dir, args.clean_dir, 
            sr=args.sample_rate, 
            window_size=args.window_size,
            lazy_load=args.lazy_load
        )
        
        if args.lazy_load:
            noisy_data, clean_data = data  # File paths
            print("Using lazy loading (will load from disk per batch)")
        else:
            noisy_data, clean_data = data  # Numpy arrays
            print(f"Dataset loaded to memory. Shape: {noisy_data.shape}")
    else:
        print("\nCreating sample data for testing...")
        noisy_data, clean_data = create_sample_data(
            num_samples=args.num_samples,
            sample_length=args.window_size,
            sr=args.sample_rate
        )
        args.lazy_load = False  # Sample data always in memory
        print(f"Sample dataset shape: {noisy_data.shape}")
    
    # Create models
    print("\nInitializing models...")
    if args.simple:
        print("Using simple models (less memory)")
        generator = SimpleGenerator(
            input_channels=1, 
            output_channels=1, 
            base_filters=args.base_filters
        )
        discriminator = SimpleDiscriminator(
            input_channels=2, 
            base_filters=args.base_filters
        )
    else:
        print("Using full models")
        generator = Generator(
            input_channels=1, 
            output_channels=1, 
            base_filters=args.base_filters
        )
        discriminator = Discriminator(
            input_channels=2, 
            base_filters=args.base_filters
        )
    
    # Print model info
    ModelSizeCalculator.print_model_info(generator, "Generator")
    ModelSizeCalculator.print_model_info(discriminator, "Discriminator")
    
    # Train
    print(f"\nStarting training with {args.gan_type}...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Lazy loading: {args.lazy_load}")
    
    trainer = train_sergan(
        train_noisy=noisy_data,
        train_clean=clean_data,
        generator=generator,
        discriminator=discriminator,
        device=device,
        gan_type=args.gan_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        lazy_load=args.lazy_load
    )
    
    # Save final model
    final_path = Path(args.save_dir) / 'final_model.pt'
    trainer.save_checkpoint(final_path, args.epochs, {})
    print(f"\nTraining completed! Final model saved to {final_path}")


def test(args):
    """Testing/Enhancement function"""
    print("="*60)
    print("SERGAN Audio Enhancement")
    print("="*60)
    
    # Get device
    device = get_device(prefer_directml=args.use_directml)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create generator
    if args.simple:
        generator = SimpleGenerator(
            input_channels=1, 
            output_channels=1,
            base_filters=args.base_filters
        )
    else:
        generator = Generator(
            input_channels=1, 
            output_channels=1,
            base_filters=args.base_filters
        )
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator = generator.to(device)
    generator.eval()
    
    print("Model loaded successfully!")
    ModelSizeCalculator.print_model_info(generator, "Generator")
    
    # Check if testing single file or testset
    if args.input:
        # Single file enhancement
        print(f"\nLoading noisy audio from {args.input}...")
        noisy_audio = load_audio(args.input, sr=args.sample_rate)
        print(f"Audio duration: {len(noisy_audio)/args.sample_rate:.2f} seconds")
        
        # Enhance
        print("\nEnhancing audio...")
        enhanced_audio = enhance_audio(
            model=generator,
            noisy_audio=noisy_audio,
            device=device,
            window_size=args.window_size,
            overlap=args.overlap
        )
        
        # Save
        print(f"Saving enhanced audio to {args.output}...")
        save_audio(enhanced_audio, args.output, sr=args.sample_rate)
        print("Enhancement completed!")
        
    elif args.test_noisy_dir:
        # Testset evaluation
        from utils import evaluate_testset
        print(f"\nEvaluating on testset...")
        print(f"Noisy dir: {args.test_noisy_dir}")
        if args.test_clean_dir:
            print(f"Clean dir: {args.test_clean_dir}")
        print(f"Output dir: {args.test_output_dir}")
        
        results = evaluate_testset(
            generator=generator,
            noisy_dir=args.test_noisy_dir,
            clean_dir=args.test_clean_dir,
            output_dir=args.test_output_dir,
            device=device,
            sr=args.sample_rate,
            window_size=args.window_size,
            overlap=args.overlap
        )
        
        print("\n" + "="*60)
        print("Test Results:")
        print("="*60)
        print(f"Total files processed: {results['total_files']}")
        if results['avg_snr_improvement'] is not None:
            print(f"Average SNR improvement: {results['avg_snr_improvement']:.2f} dB")
        print(f"Output saved to: {args.test_output_dir}")
        print("="*60)
    else:
        print("\nError: Either --input or --test-noisy-dir must be provided!")
        return


def main():
    parser = argparse.ArgumentParser(description='SERGAN PyTorch Implementation')
    
    # Mode
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                       help='Mode: train or test')
    
    # Device
    parser.add_argument('--use-directml', action='store_true',
                       help='Use DirectML for AMD/Intel GPU (Windows only)')
    
    # Model
    parser.add_argument('--simple', action='store_true',
                       help='Use simple model (less memory, faster training)')
    parser.add_argument('--base-filters', type=int, default=16,
                       help='Base number of filters')
    
    # Training
    parser.add_argument('--gan-type', type=str, default='rasgan-gp',
                       choices=['lsgan', 'wgan-gp', 'rsgan-gp', 'rasgan-gp', 'ralsgan-gp'],
                       help='GAN type')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    # Data
    parser.add_argument('--noisy-dir', type=str, default=None,
                       help='Directory with noisy audio files')
    parser.add_argument('--clean-dir', type=str, default=None,
                       help='Directory with clean audio files')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of sample data if no real data provided')
    parser.add_argument('--lazy-load', action='store_true',
                       help='Load audio from disk per batch (hemat RAM untuk dataset besar)')
    
    # Audio
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate')
    parser.add_argument('--window-size', type=int, default=16384,
                       help='Window size for processing')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio for enhancement (0-1)')
    
    # Testing
    parser.add_argument('--input', type=str, default=None,
                       help='Input noisy audio file for testing (single file)')
    parser.add_argument('--output', type=str, default='enhanced.wav',
                       help='Output enhanced audio file (single file)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint file to load')
    parser.add_argument('--test-noisy-dir', type=str, default=None,
                       help='Directory with test noisy audio files (for testset evaluation)')
    parser.add_argument('--test-clean-dir', type=str, default=None,
                       help='Directory with test clean audio files (optional, for SNR calculation)')
    parser.add_argument('--test-output-dir', type=str, default='enhanced_testset',
                       help='Output directory for enhanced test files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        if not args.checkpoint:
            parser.error("--checkpoint is required for test mode")
        if not args.input and not args.test_noisy_dir:
            parser.error("Either --input (single file) or --test-noisy-dir (testset) is required for test mode")
        test(args)


if __name__ == '__main__':
    main()