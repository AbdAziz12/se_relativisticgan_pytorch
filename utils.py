import torch
import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path

def get_device(prefer_directml=False):
    """
    Automatically detect dan pilih device yang tersedia
    
    Args:
        prefer_directml: Jika True, coba gunakan DirectML dulu
    
    Returns:
        torch.device
    """
    if prefer_directml:
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"Using DirectML device: {device}")
            return device
        except ImportError:
            print("DirectML not available, falling back to other options")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    
    device = torch.device('cpu')
    print("Using CPU device")
    return device


def load_audio(file_path, sr=16000):
    """
    Load audio file
    
    Args:
        file_path: Path ke audio file
        sr: Sample rate
    
    Returns:
        numpy array of audio samples
    """
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def save_audio(audio, file_path, sr=16000):
    """
    Save audio file
    
    Args:
        audio: numpy array atau torch tensor
        file_path: Path untuk save file
        sr: Sample rate
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, audio, sr)


def preprocess_audio(audio, window_size=16384):
    """
    Preprocess audio menjadi windows untuk training
    
    Args:
        audio: numpy array of audio
        window_size: Ukuran window
    
    Returns:
        List of audio windows
    """
    audio_length = len(audio)
    
    if audio_length < window_size:
        # Pad jika terlalu pendek
        audio = np.pad(audio, (0, window_size - audio_length))
        return [audio]
    
    # Split menjadi windows dengan overlap
    hop_size = window_size // 2
    windows = []
    
    for start in range(0, audio_length - window_size + 1, hop_size):
        window = audio[start:start + window_size]
        windows.append(window)
    
    return windows


def prepare_dataset(noisy_dir, clean_dir, sr=16000, window_size=16384, lazy_load=False):
    """
    Prepare dataset dari directory
    
    Args:
        noisy_dir: Directory berisi noisy audio files
        clean_dir: Directory berisi clean audio files
        sr: Sample rate
        window_size: Window size untuk splitting
        lazy_load: Jika True, return file paths (hemat RAM). Jika False, load semua ke memory
    
    Returns:
        Jika lazy_load=False: noisy_data, clean_data (numpy arrays)
        Jika lazy_load=True: noisy_files, clean_files (list of paths)
    """
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)
    
    noisy_files = sorted(list(noisy_dir.glob('*.wav')))
    clean_files = sorted(list(clean_dir.glob('*.wav')))
    
    print(f"Found {len(noisy_files)} noisy files and {len(clean_files)} clean files")
    
    if lazy_load:
        # Return file paths saja untuk lazy loading
        print("Using lazy loading (load per batch from disk)")
        return noisy_files, clean_files
    
    # Load semua ke memory
    print("Loading all files to memory...")
    noisy_windows = []
    clean_windows = []
    
    for noisy_file, clean_file in zip(noisy_files, clean_files):
        # Load audio
        noisy_audio = load_audio(str(noisy_file), sr)
        clean_audio = load_audio(str(clean_file), sr)
        
        # Pastikan panjangnya sama
        min_len = min(len(noisy_audio), len(clean_audio))
        noisy_audio = noisy_audio[:min_len]
        clean_audio = clean_audio[:min_len]
        
        # Split menjadi windows
        noisy_wins = preprocess_audio(noisy_audio, window_size)
        clean_wins = preprocess_audio(clean_audio, window_size)
        
        noisy_windows.extend(noisy_wins)
        clean_windows.extend(clean_wins)
    
    print(f"Total windows: {len(noisy_windows)}")
    
    return np.array(noisy_windows), np.array(clean_windows)


def enhance_audio(model, noisy_audio, device, window_size=16384, overlap=0.5):
    """
    Enhance audio menggunakan trained model
    
    Args:
        model: Trained generator model
        noisy_audio: numpy array of noisy audio
        device: torch device
        window_size: Window size
        overlap: Overlap ratio (0-1)
    
    Returns:
        Enhanced audio (numpy array)
    """
    model.eval()
    
    audio_length = len(noisy_audio)
    hop_size = int(window_size * (1 - overlap))
    
    # Output buffer
    enhanced = np.zeros(audio_length)
    window_sum = np.zeros(audio_length)
    
    # Hanning window untuk smooth overlap
    hann_window = np.hanning(window_size)
    
    with torch.no_grad():
        for start in range(0, audio_length - window_size + 1, hop_size):
            # Extract window
            window = noisy_audio[start:start + window_size]
            
            # Convert ke tensor
            window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
            
            # Enhance
            enhanced_window = model(window_tensor)
            enhanced_window = enhanced_window.squeeze().cpu().numpy()
            
            # Apply hanning window dan accumulate
            enhanced[start:start + window_size] += enhanced_window * hann_window
            window_sum[start:start + window_size] += hann_window
        
        # Handle last window jika tidak sempurna
        if audio_length % hop_size != 0:
            start = audio_length - window_size
            window = noisy_audio[start:]
            if len(window) < window_size:
                window = np.pad(window, (0, window_size - len(window)))
            
            window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
            enhanced_window = model(window_tensor).squeeze().cpu().numpy()
            
            enhanced[start:] += enhanced_window[:len(enhanced[start:])] * hann_window[:len(enhanced[start:])]
            window_sum[start:] += hann_window[:len(window_sum[start:])]
    
    # Normalize by window sum
    enhanced = np.divide(enhanced, window_sum, where=window_sum != 0)
    
    return enhanced


def calculate_snr(clean, enhanced):
    """
    Calculate Signal-to-Noise Ratio
    
    Args:
        clean: Clean audio signal
        enhanced: Enhanced audio signal
    
    Returns:
        SNR in dB
    """
    if isinstance(clean, torch.Tensor):
        clean = clean.cpu().numpy()
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.cpu().numpy()
    
    noise = clean - enhanced
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def normalize_audio(audio):
    """
    Normalize audio ke range [-1, 1]
    
    Args:
        audio: Audio signal
    
    Returns:
        Normalized audio
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


class ModelSizeCalculator:
    """Helper untuk menghitung ukuran model"""
    
    @staticmethod
    def count_parameters(model):
        """Count total parameters"""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def count_trainable_parameters(model):
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size_mb(model):
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    @staticmethod
    def print_model_info(model, model_name="Model"):
        """Print model information"""
        total_params = ModelSizeCalculator.count_parameters(model)
        trainable_params = ModelSizeCalculator.count_trainable_parameters(model)
        size_mb = ModelSizeCalculator.get_model_size_mb(model)
        
        print(f"\n{'='*50}")
        print(f"{model_name} Information:")
        print(f"{'='*50}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {size_mb:.2f} MB")
        print(f"{'='*50}\n")


def create_sample_data(num_samples=100, sample_length=16384, sr=16000):
    """
    Create sample noisy/clean data untuk testing
    
    Args:
        num_samples: Number of samples
        sample_length: Length of each sample
        sr: Sample rate
    
    Returns:
        noisy_data, clean_data
    """
    print(f"Creating {num_samples} sample audio pairs...")
    
    clean_data = []
    noisy_data = []
    
    for _ in range(num_samples):
        # Generate clean signal (sine wave dengan random frequency)
        t = np.linspace(0, sample_length/sr, sample_length)
        freq = np.random.uniform(200, 2000)
        clean = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Add noise
        noise = np.random.normal(0, 0.1, sample_length)
        noisy = clean + noise
        
        clean_data.append(clean)
        noisy_data.append(noisy)
    
    return np.array(noisy_data), np.array(clean_data)


def evaluate_testset(generator, noisy_dir, clean_dir, output_dir, 
                     device, sr=16000, window_size=16384, overlap=0.5):
    """
    Evaluate model pada testset
    
    Args:
        generator: Trained generator model
        noisy_dir: Directory dengan noisy test files
        clean_dir: Directory dengan clean test files (optional untuk SNR calc)
        output_dir: Directory untuk save enhanced files
        device: torch device
        sr: Sample rate
        window_size: Window size
        overlap: Overlap ratio
    
    Returns:
        Dictionary dengan evaluation results
    """
    from pathlib import Path
    import os
    
    noisy_dir = Path(noisy_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    noisy_files = sorted(list(noisy_dir.glob('*.wav')))
    
    if not noisy_files:
        print(f"Warning: No .wav files found in {noisy_dir}")
        return {'total_files': 0, 'avg_snr_improvement': None}
    
    print(f"Found {len(noisy_files)} test files")
    
    # Check if clean dir provided
    calculate_snr_flag = False
    if clean_dir:
        clean_dir = Path(clean_dir)
        if clean_dir.exists():
            calculate_snr_flag = True
            print("Clean reference files found - will calculate SNR improvement")
    
    snr_improvements = []
    processed_files = 0
    
    generator.eval()
    with torch.no_grad():
        for noisy_file in noisy_files:
            try:
                print(f"Processing: {noisy_file.name}")
                
                # Load noisy audio
                noisy_audio = load_audio(str(noisy_file), sr=sr)
                
                # Enhance
                enhanced_audio = enhance_audio(
                    model=generator,
                    noisy_audio=noisy_audio,
                    device=device,
                    window_size=window_size,
                    overlap=overlap
                )
                
                # Save enhanced
                output_path = output_dir / noisy_file.name
                save_audio(enhanced_audio, str(output_path), sr=sr)
                
                # Calculate SNR if clean reference available
                if calculate_snr_flag:
                    clean_file = clean_dir / noisy_file.name
                    if clean_file.exists():
                        clean_audio = load_audio(str(clean_file), sr=sr)
                        
                        # Ensure same length
                        min_len = min(len(clean_audio), len(enhanced_audio))
                        clean_audio = clean_audio[:min_len]
                        enhanced_audio_trimmed = enhanced_audio[:min_len]
                        noisy_audio_trimmed = noisy_audio[:min_len]
                        
                        # Calculate SNR
                        snr_noisy = calculate_snr(clean_audio, noisy_audio_trimmed)
                        snr_enhanced = calculate_snr(clean_audio, enhanced_audio_trimmed)
                        snr_improvement = snr_enhanced - snr_noisy
                        
                        snr_improvements.append(snr_improvement)
                        print(f"  SNR improvement: {snr_improvement:.2f} dB")
                
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {noisy_file.name}: {e}")
                continue
    
    # Calculate average SNR improvement
    avg_snr_improvement = None
    if snr_improvements:
        avg_snr_improvement = np.mean(snr_improvements)
    
    return {
        'total_files': processed_files,
        'avg_snr_improvement': avg_snr_improvement,
        'snr_improvements': snr_improvements
    }