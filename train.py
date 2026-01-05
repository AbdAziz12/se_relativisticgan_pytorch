import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import librosa
import torch.nn.functional as F

from config import Config

class AudioDataset(Dataset):
    """Dataset untuk audio noisy dan clean (data sudah di memory)"""
    def __init__(self, noisy_data, clean_data, window_size=16384):
        self.noisy_data = noisy_data
        self.clean_data = clean_data
        self.window_size = window_size
        
    def __len__(self):
        return len(self.noisy_data)
    
    def __getitem__(self, idx):
        noisy = torch.FloatTensor(self.noisy_data[idx]).unsqueeze(0)  # [1, T]
        clean = torch.FloatTensor(self.clean_data[idx]).unsqueeze(0)  # [1, T]
        
        # Pad jika perlu
        if noisy.shape[1] < self.window_size:
            pad_size = self.window_size - noisy.shape[1]
            noisy = torch.nn.functional.pad(noisy, (0, pad_size))
            clean = torch.nn.functional.pad(clean, (0, pad_size))
        elif noisy.shape[1] > self.window_size:
            noisy = noisy[:, :self.window_size]
            clean = clean[:, :self.window_size]
            
        return noisy, clean


class LazyAudioDataset(Dataset):
    """
    Dataset yang load audio dari disk per batch (hemat RAM)
    Cocok untuk dataset besar
    """
    def __init__(self, noisy_files, clean_files, sr=16000, window_size=16384,
                 apply_preemph=False, preemph_coeff=0.95):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.sr = sr
        self.window_size = window_size
        self.apply_preemph = apply_preemph
        self.preemph_coeff = preemph_coeff
        
        # Precompute jumlah windows per file
        self.file_windows = []
        self.cumsum_windows = [0]
        
        print("Analyzing dataset...")
        for nf, cf in zip(noisy_files, clean_files):
            # Load untuk hitung panjang
            noisy, _ = librosa.load(nf, sr=sr)
            n_windows = max(1, len(noisy) // (window_size // 2))
            self.file_windows.append(n_windows)
            self.cumsum_windows.append(self.cumsum_windows[-1] + n_windows)
        
        print(f"Total windows: {self.cumsum_windows[-1]}")
    
    def __len__(self):
        return self.cumsum_windows[-1]
    
    def __getitem__(self, idx):
        # Find which file this index belongs to
        file_idx = 0
        for i, cumsum in enumerate(self.cumsum_windows[1:]):
            if idx < cumsum:
                file_idx = i
                break
        
        window_idx = idx - self.cumsum_windows[file_idx]
        
        # Load audio files
        noisy, _ = librosa.load(str(self.noisy_files[file_idx]), sr=self.sr)
        clean, _ = librosa.load(str(self.clean_files[file_idx]), sr=self.sr)

        # Apply preemphasis if enabled
        if self.apply_preemph:
            from utils import pre_emph
            noisy = pre_emph(noisy, coeff=self.preemph_coeff)
            clean = pre_emph(clean, coeff=self.preemph_coeff)
        
        # Pastikan panjangnya sama
        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]
        
        # Extract window
        start = window_idx * (self.window_size // 2)
        end = start + self.window_size
        
        noisy_window = noisy[start:end]
        clean_window = clean[start:end]
        
        # Pad if necessary
        if len(noisy_window) < self.window_size:
            noisy_window = np.pad(noisy_window, (0, self.window_size - len(noisy_window)))
            clean_window = np.pad(clean_window, (0, self.window_size - len(clean_window)))
        
        noisy_tensor = torch.FloatTensor(noisy_window).unsqueeze(0)
        clean_tensor = torch.FloatTensor(clean_window).unsqueeze(0)
        
        return noisy_tensor, clean_tensor

# train.py - tambahkan di atas class SERGANTrainer
class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss untuk speech enhancement
    """
    def __init__(self, fft_sizes=[512, 256, 128, 64], hop_ratio=0.25, win_ratio=1.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_ratio = hop_ratio
        self.win_ratio = win_ratio
        
    def forward(self, x, y):
        """
        x: generated speech [batch, 1, samples]
        y: clean speech [batch, 1, samples]
        """
        total_loss = 0.0
        
        for fft_size in self.fft_sizes:
            hop_size = int(fft_size * self.hop_ratio)
            win_size = int(fft_size * self.win_ratio)
            
            # Pastikan panjang cukup untuk STFT
            if x.size(-1) < win_size:
                x_padded = F.pad(x, (0, win_size - x.size(-1)))
                y_padded = F.pad(y, (0, win_size - y.size(-1)))
            else:
                x_padded = x
                y_padded = y
            
            # STFT
            window = torch.hann_window(win_size, device=x.device)
            
            X = torch.stft(
                x_padded.squeeze(1), 
                n_fft=fft_size, 
                hop_length=hop_size, 
                win_length=win_size,
                window=window, 
                return_complex=True, 
                center=False
            )
            
            Y = torch.stft(
                y_padded.squeeze(1),
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                window=window,
                return_complex=True,
                center=False
            )
            
            # Magnitude spectra
            mag_X = torch.abs(X)
            mag_Y = torch.abs(Y)
            
            # Log magnitude (lebih robust)
            log_mag_X = torch.log(mag_X + 1e-7)
            log_mag_Y = torch.log(mag_Y + 1e-7)
            
            # L1 loss pada log magnitude
            loss_mag = F.l1_loss(log_mag_X, log_mag_Y)
            
            total_loss += loss_mag
        
        return total_loss / len(self.fft_sizes)

class SERGANTrainer:
    """
    Trainer untuk SERGAN dengan berbagai loss functions:
    - LSGAN: Least Squares GAN
    - WGAN-GP: Wasserstein GAN with Gradient Penalty
    - RSGAN-GP: Relativistic Standard GAN with GP
    - RaSGAN-GP: Relativistic average Standard GAN with GP
    - RaLSGAN-GP: Relativistic average Least Squares GAN with GP
    """
    
    def __init__(self, generator, discriminator, device, gan_type='rasgan-gp',
                 use_spec_loss=True, spec_loss_weight=1.0):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.gan_type = gan_type.lower()
        self.use_spec_loss = use_spec_loss
        self.spec_loss_weight = spec_loss_weight
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))
        
        # Loss functions
        self.l1_loss = nn.L1Loss()

        # Spectral loss (hanya untuk training, BUKAN bagian model)
        if self.use_spec_loss:
            self.spec_loss_fn = MultiResolutionSTFTLoss().to(device)
        
    def gradient_penalty(self, real_data, fake_data, noisy_data):
        """Compute gradient penalty untuk WGAN-GP"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_(True)
        
        # Concatenate dengan noisy
        d_interpolates_input = torch.cat([interpolates, noisy_data], dim=1)
        d_interpolates = self.discriminator(d_interpolates_input)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train_step(self, noisy, clean):
        """Single training step"""
        batch_size = noisy.size(0)
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)
        
        # =================== Train Discriminator ===================
        self.d_optimizer.zero_grad()
        
        # Generate fake
        with torch.no_grad():
            fake = self.generator(noisy)
        
        # Discriminator outputs
        real_input = torch.cat([clean, noisy], dim=1)
        fake_input = torch.cat([fake.detach(), noisy], dim=1)
        
        d_real = self.discriminator(real_input)
        d_fake = self.discriminator(fake_input)
        
        # Compute discriminator loss berdasarkan gan_type
        if self.gan_type == 'lsgan':
            d_loss_real = torch.mean((d_real - 1) ** 2)
            d_loss_fake = torch.mean(d_fake ** 2)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
        elif self.gan_type == 'wgan-gp':
            gp = self.gradient_penalty(clean, fake.detach(), noisy)
            d_loss = -torch.mean(d_real) + torch.mean(d_fake) + 10 * gp
            
        elif self.gan_type == 'rsgan-gp':
            gp = self.gradient_penalty(clean, fake.detach(), noisy)
            # d_loss = torch.mean((d_real - d_fake - 1) ** 2) + 10 * gp SALAH BOI
            d_loss = -torch.mean(F.logsigmoid(d_real - d_fake)) + 10 * gp
            
        elif self.gan_type == 'rasgan-gp':
            d_real_mean = torch.mean(d_real)
            d_fake_mean = torch.mean(d_fake)
            gp = self.gradient_penalty(clean, fake.detach(), noisy)
            # d_loss = torch.mean((d_real - d_fake_mean - 1) ** 2) + \
            #          torch.mean((d_fake - d_real_mean + 1) ** 2) + 10 * gp
            d_loss = -torch.mean(F.logsigmoid(d_real - d_fake_mean)) - \
                    torch.mean(F.logsigmoid(-(d_fake - d_real_mean))) + 10 * gp
            
        elif self.gan_type == 'ralsgan-gp':
            d_real_mean = torch.mean(d_real)
            d_fake_mean = torch.mean(d_fake)
            gp = self.gradient_penalty(clean, fake.detach(), noisy)
            d_loss = torch.mean((d_real - d_fake_mean - 1) ** 2) + \
                     torch.mean((d_fake - d_real_mean + 1) ** 2) + 10 * gp
        else:
            raise ValueError(f"Unknown GAN type: {self.gan_type}")
        
        d_loss.backward()
        self.d_optimizer.step()
        
        # =================== Train Generator ===================
        self.g_optimizer.zero_grad()
        
        # Generate fake
        fake = self.generator(noisy)
        
        # Discriminator outputs
        fake_input_gen = torch.cat([fake, noisy], dim=1)
        real_input_gen = torch.cat([clean, noisy], dim=1)

        with torch.no_grad():
            d_real_gen = self.discriminator(real_input_gen)
        
        d_fake_gen = self.discriminator(fake_input_gen)

        
        # Compute generator loss
        l1_loss = self.l1_loss(fake, clean) * 100  # L1 loss weight

        # Spectral loss jika diaktifkan
        spec_loss_value = 0.0
        if self.use_spec_loss:
            spec_loss_value = self.spec_loss_fn(fake, clean) * self.spec_loss_weight
        
        if self.gan_type == 'lsgan':
            g_loss_adv = torch.mean((d_fake_gen - 1) ** 2)
            
        elif self.gan_type == 'wgan-gp':
            g_loss_adv = -torch.mean(d_fake_gen)
            
        elif self.gan_type == 'rsgan-gp':
            # d_real = self.discriminator(real_input)
            # g_loss_adv = torch.mean((d_fake_gen - d_real_gen + 1) ** 2)
            g_loss_adv = -torch.mean(F.logsigmoid(d_fake_gen - d_real_gen))
            
        elif self.gan_type == 'rasgan-gp':
            # d_real = self.discriminator(real_input)
            d_real_mean_gen = torch.mean(d_real_gen)
            d_fake_mean_gen = torch.mean(d_fake_gen)
            # g_loss_adv = torch.mean((d_fake - d_real_mean - 1) ** 2) + \
            #              torch.mean((d_real - d_fake_mean + 1) ** 2)
            g_loss_adv = -torch.mean(F.logsigmoid(d_fake_gen - d_real_mean_gen)) - \
                        torch.mean(F.logsigmoid(-(d_real_gen - d_fake_mean_gen)))
            
        elif self.gan_type == 'ralsgan-gp':
            # d_real = self.discriminator(real_input)
            d_real_mean_gen = torch.mean(d_real_gen)
            d_fake_mean_gen = torch.mean(d_fake_gen)
            g_loss_adv = torch.mean((d_fake_gen - d_real_mean_gen - 1) ** 2) + \
                         torch.mean((d_real_gen - d_fake_mean_gen + 1) ** 2)
        
        # Ada spectral loss
        g_loss = g_loss_adv + l1_loss + spec_loss_value
        g_loss.backward()
        self.g_optimizer.step()
        
        metrics = {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_loss_adv': g_loss_adv.item(),
            'l1_loss': l1_loss.item()
        }

        if self.use_spec_loss:
            metrics['spec_loss'] = spec_loss_value.item()
        
        return metrics
    
    def train_epoch(self, dataloader, epoch):
        """Train satu epoch"""
        self.generator.train()
        self.discriminator.train()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        metrics = {'d_loss': 0, 'g_loss': 0, 'g_loss_adv': 0, 'l1_loss': 0}
        
        if self.use_spec_loss:
            metrics['spec_loss'] = 0

        for i, (noisy, clean) in enumerate(pbar):
            losses = self.train_step(noisy, clean)
            
            for key in metrics:
                metrics[key] += losses[key]
            
            # Update progress bar
            avg_metrics = {k: v/(i+1) for k, v in metrics.items()}
            pbar.set_postfix(avg_metrics)
        
        return {k: v/len(dataloader) for k, v in metrics.items()}
    
    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'metrics': metrics
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['metrics']


def train_sergan(train_noisy, train_clean, generator, discriminator, 
                 device, gan_type='rasgan-gp', epochs=100, batch_size=4,
                 save_dir='checkpoints', lazy_load=False,
                 apply_preemph=False, preemph_coeff=0.95,
                 use_spec_loss=True, spec_loss_weight=5.0,
                 checkpoint_path=None):
    """
    Main training function
    
    Args:
        train_noisy: numpy array of noisy audio ATAU list of file paths (jika lazy_load=True)
        train_clean: numpy array of clean audio ATAU list of file paths (jika lazy_load=True)
        generator: Generator model
        discriminator: Discriminator model
        device: torch device
        gan_type: Type of GAN ('lsgan', 'wgan-gp', 'rsgan-gp', 'rasgan-gp', 'ralsgan-gp')
        epochs: Number of training epochs
        batch_size: Batch size
        save_dir: Directory to save checkpoints
        lazy_load: Jika True, train_noisy dan train_clean adalah file paths
    """
    
    # Create dataset and dataloader
    if lazy_load:
        print("Using LazyAudioDataset (load from disk per batch)")
        dataset = LazyAudioDataset(train_noisy, train_clean,
                                   apply_preemph=apply_preemph,
                                   preemph_coeff=preemph_coeff)
    else:
        print("Using AudioDataset (all data in memory)")
        dataset = AudioDataset(train_noisy, train_clean)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Create trainer
    trainer = SERGANTrainer(generator, discriminator, device, gan_type,
                            use_spec_loss=use_spec_loss, spec_loss_weight=spec_loss_weight)
    
    # Training loop
    start_epoch = 1
    if checkpoint_path:
        print(f"📂 Loading checkpoint from {checkpoint_path}...")
        loaded_epoch, loaded_metrics = trainer.load_checkpoint(checkpoint_path)
        start_epoch = loaded_epoch + 1
        print(f"🔄 Resuming training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, epochs + 1):  # 👈 Mulai dari start_epoch
        metrics = trainer.train_epoch(dataloader, epoch)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"D Loss: {metrics['d_loss']:.4f}")
        print(f"G Loss: {metrics['g_loss']:.4f}")
        print(f"G Adv Loss: {metrics['g_loss_adv']:.4f}")
        print(f"L1 Loss: {metrics['l1_loss']:.4f}")
        if use_spec_loss:
            print(f"Spec Loss: {metrics.get('spec_loss', 0):.4f}")
        
        # Save checkpoint setiap N epoch
        if epoch % Config.SAVE_EVERY_N_EPOCHS == 0 or epoch == epochs:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            trainer.save_checkpoint(checkpoint_path, epoch, metrics)
    
    return trainer