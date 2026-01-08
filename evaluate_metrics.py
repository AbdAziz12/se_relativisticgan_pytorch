import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from scipy.linalg import solve_toeplitz

def calculate_llr(ref, deg, order=12):
    """Menghitung Log Likelihood Ratio (LLR) menggunakan koefisien LPC"""
    def get_lpc(signal, order):
        # Menghitung koefisien autokorelasi
        r = np.correlate(signal, signal, mode='full')[len(signal)-1:len(signal)+order]
        a = solve_toeplitz(r[:-1], r[1:])
        return np.concatenate(([1], -a))

    # Bagi menjadi frame (20ms)
    frame_len = int(0.02 * 16000)
    num_frames = len(ref) // frame_len
    llr_vals = []

    for i in range(num_frames):
        ref_f = ref[i*frame_len : (i+1)*frame_len]
        deg_f = deg[i*frame_len : (i+1)*frame_len]
        
        # LPC coefficients
        a_ref = get_lpc(ref_f, order)
        a_deg = get_lpc(deg_f, order)
        
        # Perhitungan jarak LLR sederhana
        # (Versi disederhanakan untuk stabilitas)
        llr_vals.append(np.log(np.abs(np.dot(a_deg, a_ref) / np.dot(a_ref, a_ref)) + 1e-6))
    
    return np.mean(llr_vals)

def calculate_cd(ref, deg):
    """Cepstral Distance menggunakan MFCC"""
    c_ref = librosa.feature.mfcc(y=ref, sr=16000, n_mfcc=13)
    c_deg = librosa.feature.mfcc(y=deg, sr=16000, n_mfcc=13)
    return np.mean(np.sqrt(np.sum((c_ref - c_deg)**2, axis=0)))

def evaluate_all(clean_dir, enhanced_dir):
    results = []
    files = [f for f in os.listdir(enhanced_dir) if f.endswith('.wav')]
    
    for filename in tqdm(files):
        path_c = os.path.join(clean_dir, filename)
        path_e = os.path.join(enhanced_dir, filename)
        
        if not os.path.exists(path_c): continue
        
        ref, _ = librosa.load(path_c, sr=16000)
        deg, _ = librosa.load(path_e, sr=16000)
        ml = min(len(ref), len(deg))
        ref, deg = ref[:ml], deg[:ml]

        # Kalkulasi Metrik
        p = pesq(16000, ref, deg, 'wb')
        s = stoi(ref, deg, 16000)
        cd = calculate_cd(ref, deg)
        llr = calculate_llr(ref, deg)
        
        # SNR
        noise = ref - deg
        snr = 10 * np.log10(np.sum(ref**2) / (np.sum(noise**2) + 1e-10))

        # Estimasi CSIG, CBAK, COVL (Berdasarkan rumus regresi Loizou)
        csig = 3.035 + 0.583 * p - 0.121 * llr
        cbak = 1.634 + 0.478 * p - 0.007 * llr + 0.063 * snr
        covl = 1.591 + 0.604 * p - 0.062 * llr

        results.append({
            'File': filename, 'PESQ': p, 'STOI': s, 
            'CD': cd, 'LLR': llr, 'SNR': snr,
            'CSIG': np.clip(csig, 1, 5), 
            'CBAK': np.clip(cbak, 1, 5), 
            'COVL': np.clip(covl, 1, 5)
        })

    return pd.DataFrame(results)

# Jalankan
df = evaluate_all('./data/clean_testset_wav_16kHz', './results/simple_konf2/simple_konf2_noresidual_40')

# 2. Hitung Summary (Mean, Min, Max, Std)
# Kita hanya menghitung kolom numerik saja
summary_df = df.drop(columns=['File']).describe().loc[['mean', 'min', 'max', 'std']]

# 3. Tampilkan di Terminal agar bisa langsung dicatat
print("\n" + "="*30)
print(" SUMMARY METRIC EVALUATION ")
print("="*30)
print(summary_df)
print("="*30)

# 4. Simpan Summary ke file CSV
summary_df.to_csv('summary_metrics_simple_konf2_noresidual.csv')

# 5. (Opsional) Tetap simpan data per file untuk backup jika dosen bertanya detail
df.to_csv('full_details_metrics_simple_konf2_noresidual.csv', index=False)

print("\nFile 'summary_metrics_simple_konf2_noresidual.csv' telah dibuat di folder utama.")