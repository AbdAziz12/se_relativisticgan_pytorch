import torch
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path


class CheckpointExtractor:
    """
    Ekstraksi parameter dari PyTorch checkpoint (.pt) untuk implementasi FPGA
    """
    
    def __init__(self, checkpoint_path):
        """
        Args:
            checkpoint_path: Path ke file checkpoint .pt
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.output_dir = Path(checkpoint_path).parent / 'extracted_params'
        self.output_dir.mkdir(exist_ok=True)
        
    def get_model_info(self):
        """Dapatkan informasi umum tentang checkpoint"""
        info = {
            'checkpoint_file': str(self.checkpoint_path),
            'epoch': self.checkpoint.get('epoch', 'N/A'),
            'keys': list(self.checkpoint.keys())
        }
        
        # Hitung total parameter untuk generator dan discriminator
        if 'generator_state_dict' in self.checkpoint:
            gen_params = sum(p.numel() for p in self.checkpoint['generator_state_dict'].values())
            info['generator_total_params'] = gen_params
            
        if 'discriminator_state_dict' in self.checkpoint:
            disc_params = sum(p.numel() for p in self.checkpoint['discriminator_state_dict'].values())
            info['discriminator_total_params'] = disc_params
            
        return info
    
    def extract_layer_to_csv(self, tensor, layer_name, model_type='generator'):
        """
        Ekstraksi satu layer ke CSV
        
        Args:
            tensor: PyTorch tensor
            layer_name: Nama layer (akan dijadikan nama file)
            model_type: 'generator' atau 'discriminator'
        """
        # Buat direktori untuk model
        model_dir = self.output_dir / model_type / 'csv'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert tensor ke numpy
        data = tensor.detach().cpu().numpy()
        
        # Sanitize nama file
        safe_name = layer_name.replace('.', '_').replace('/', '_')
        filepath = model_dir / f"{safe_name}.csv"
        
        # Simpan berdasarkan dimensi tensor
        if len(data.shape) == 1:  # Bias (1D)
            df = pd.DataFrame({
                'index': range(len(data)),
                'value': data
            })
            df.to_csv(filepath, index=False)
            
        elif len(data.shape) == 2:  # Weight matrix (2D)
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
        elif len(data.shape) == 3:  # Conv1D weights (out_ch, in_ch, kernel)
            # Simpan sebagai beberapa CSV (satu per output channel)
            for i in range(data.shape[0]):
                df = pd.DataFrame(data[i])
                out_file = model_dir / f"{safe_name}_outch{i}.csv"
                df.to_csv(out_file, index=False)
                
        elif len(data.shape) == 4:  # Conv2D weights (out_ch, in_ch, h, w)
            # Simpan sebagai beberapa CSV (satu per output channel)
            for i in range(data.shape[0]):
                # Reshape ke 2D untuk setiap output channel
                reshaped = data[i].reshape(data.shape[1], -1)
                df = pd.DataFrame(reshaped)
                out_file = model_dir / f"{safe_name}_outch{i}.csv"
                df.to_csv(out_file, index=False)
        
        return filepath
    
    def extract_layer_to_binary(self, tensor, layer_name, model_type='generator', dtype='float32'):
        """
        Ekstraksi layer ke format binary (untuk FPGA)
        
        Args:
            tensor: PyTorch tensor
            layer_name: Nama layer
            model_type: 'generator' atau 'discriminator'
            dtype: 'float32', 'float16', atau 'int16' (fixed-point)
        """
        model_dir = self.output_dir / model_type / 'binary'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name = layer_name.replace('.', '_').replace('/', '_')
        
        # Convert tensor ke numpy
        data = tensor.detach().cpu().numpy()
        
        if dtype == 'float32':
            data = data.astype(np.float32)
            ext = 'f32'
        elif dtype == 'float16':
            data = data.astype(np.float16)
            ext = 'f16'
        elif dtype == 'int16':
            # Convert ke fixed-point Q15 format (16-bit signed)
            # Range [-1, 1] -> [-32768, 32767]
            data = np.clip(data, -1.0, 1.0)
            data = (data * 32767).astype(np.int16)
            ext = 'q15'
        
        filepath = model_dir / f"{safe_name}.{ext}.bin"
        data.tofile(filepath)
        
        # Simpan metadata
        meta = {
            'shape': list(data.shape),
            'dtype': str(data.dtype),
            'size_bytes': data.nbytes,
            'min_value': float(data.min()),
            'max_value': float(data.max()),
            'mean_value': float(data.mean())
        }
        
        meta_file = model_dir / f"{safe_name}.{ext}.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        return filepath
    
    def extract_all_layers(self, model_type='generator', formats=['csv', 'binary']):
        """
        Ekstraksi semua layer dari generator atau discriminator
        
        Args:
            model_type: 'generator' atau 'discriminator'
            formats: List format output ['csv', 'binary', 'both']
        """
        # Tentukan state_dict yang akan diekstrak
        if model_type == 'generator':
            state_dict_key = 'generator_state_dict'
        elif model_type == 'discriminator':
            state_dict_key = 'discriminator_state_dict'
        else:
            raise ValueError("model_type harus 'generator' atau 'discriminator'")
        
        if state_dict_key not in self.checkpoint:
            print(f"❌ {state_dict_key} tidak ditemukan dalam checkpoint!")
            return
        
        state_dict = self.checkpoint[state_dict_key]
        
        print(f"\n📦 Mengekstrak {model_type.upper()}...")
        print(f"Total layers: {len(state_dict)}")
        
        # Buat summary
        summary = []
        
        for layer_name, tensor in state_dict.items():
            layer_info = {
                'layer_name': layer_name,
                'shape': str(tuple(tensor.shape)),
                'num_params': tensor.numel(),
                'dtype': str(tensor.dtype),
                'min': float(tensor.min()),
                'max': float(tensor.max()),
                'mean': float(tensor.mean()),
                'std': float(tensor.std())
            }
            
            # Ekstraksi ke CSV
            if 'csv' in formats or 'both' in formats:
                self.extract_layer_to_csv(tensor, layer_name, model_type)
                
            # Ekstraksi ke Binary
            if 'binary' in formats or 'both' in formats:
                # Simpan dalam 3 format: float32, float16, int16
                self.extract_layer_to_binary(tensor, layer_name, model_type, 'float32')
                self.extract_layer_to_binary(tensor, layer_name, model_type, 'float16')
                self.extract_layer_to_binary(tensor, layer_name, model_type, 'int16')
            
            summary.append(layer_info)
            print(f"  ✓ {layer_name}: {tuple(tensor.shape)} ({tensor.numel()} params)")
        
        # Simpan summary sebagai CSV (always works)
        df_summary = pd.DataFrame(summary)
        summary_dir = self.output_dir / model_type
        df_summary.to_csv(summary_dir / f'{model_type}_summary.csv', index=False)
        
        # Try to save as Excel if openpyxl is available
        try:
            df_summary.to_excel(summary_dir / f'{model_type}_summary.xlsx', index=False)
            print(f"  📊 Excel summary: {model_type}_summary.xlsx")
        except ImportError:
            print(f"  ⚠️  openpyxl tidak terinstall, skip Excel export")
            print(f"     Install dengan: pip install openpyxl")
        
        print(f"  📊 CSV summary: {model_type}_summary.csv")
        print(f"\n✅ {model_type.upper()} berhasil diekstrak!")
        print(f"📁 Output directory: {summary_dir}")
        
        return df_summary
    
    def create_architecture_report(self):
        """
        Buat laporan arsitektur lengkap untuk dokumentasi FPGA
        """
        report = {
            'checkpoint_info': self.get_model_info(),
            'generator_architecture': {},
            'discriminator_architecture': {}
        }
        
        # Analisis generator
        if 'generator_state_dict' in self.checkpoint:
            gen_dict = self.checkpoint['generator_state_dict']
            report['generator_architecture'] = self._analyze_architecture(gen_dict)
        
        # Analisis discriminator
        if 'discriminator_state_dict' in self.checkpoint:
            disc_dict = self.checkpoint['discriminator_state_dict']
            report['discriminator_architecture'] = self._analyze_architecture(disc_dict)
        
        # Simpan ke JSON
        report_file = self.output_dir / 'architecture_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Laporan arsitektur disimpan ke: {report_file}")
        
        return report
    
    def _analyze_architecture(self, state_dict):
        """Analisis arsitektur dari state_dict"""
        architecture = {
            'total_layers': len(state_dict),
            'total_parameters': sum(t.numel() for t in state_dict.values()),
            'layers': []
        }
        
        for name, tensor in state_dict.items():
            layer_type = 'unknown'
            if 'conv' in name.lower():
                layer_type = 'convolution'
            elif 'linear' in name.lower() or 'fc' in name.lower():
                layer_type = 'fully_connected'
            elif 'bn' in name.lower() or 'batch_norm' in name.lower():
                layer_type = 'batch_norm'
            elif 'bias' in name.lower():
                layer_type = 'bias'
            elif 'weight' in name.lower():
                layer_type = 'weight'
            
            layer_info = {
                'name': name,
                'type': layer_type,
                'shape': list(tensor.shape),
                'parameters': int(tensor.numel()),
                'memory_mb': float(tensor.numel() * 4 / 1024 / 1024)  # Assuming float32
            }
            
            architecture['layers'].append(layer_info)
        
        return architecture
    
    def export_for_verilog(self, model_type='generator', layer_name=None):
        """
        Ekspor parameter dalam format yang mudah digunakan untuk Verilog/VHDL
        Format: hexadecimal untuk fixed-point representation
        """
        verilog_dir = self.output_dir / model_type / 'verilog'
        verilog_dir.mkdir(parents=True, exist_ok=True)
        
        if model_type == 'generator':
            state_dict = self.checkpoint['generator_state_dict']
        else:
            state_dict = self.checkpoint['discriminator_state_dict']
        
        # Jika layer_name spesifik diberikan
        if layer_name:
            if layer_name not in state_dict:
                print(f"❌ Layer {layer_name} tidak ditemukan!")
                return
            layers = {layer_name: state_dict[layer_name]}
        else:
            layers = state_dict
        
        for name, tensor in layers.items():
            safe_name = name.replace('.', '_').replace('/', '_')
            data = tensor.detach().cpu().numpy().flatten()
            
            # Convert ke fixed-point Q15
            data_q15 = np.clip(data, -1.0, 1.0)
            data_q15 = (data_q15 * 32767).astype(np.int16)
            
            # Export sebagai hex file untuk Verilog $readmemh
            hex_file = verilog_dir / f"{safe_name}.hex"
            with open(hex_file, 'w') as f:
                for val in data_q15:
                    # Convert int16 ke unsigned 16-bit dengan proper handling
                    # Use numpy's uint16 for proper conversion
                    unsigned_val = np.uint16(val)
                    f.write(f"{unsigned_val:04X}\n")
            
            # Buat Verilog ROM module template
            verilog_file = verilog_dir / f"{safe_name}_rom.v"
            self._generate_verilog_rom(verilog_file, safe_name, len(data_q15))
            
            print(f"  ✓ {name} -> {hex_file.name}")
    
    def _generate_verilog_rom(self, filepath, module_name, size):
        """Generate Verilog ROM module template"""
        addr_width = int(np.ceil(np.log2(size)))
        
        verilog_code = f"""// Auto-generated ROM module for {module_name}
module {module_name}_rom (
    input wire clk,
    input wire [{addr_width-1}:0] addr,
    output reg [15:0] data
);

    reg [15:0] rom [0:{size-1}];
    
    initial begin
        $readmemh("{module_name}.hex", rom);
    end
    
    always @(posedge clk) begin
        data <= rom[addr];
    end

endmodule
"""
        
        with open(filepath, 'w') as f:
            f.write(verilog_code)


def main():
    """
    Contoh penggunaan
    """
    # Path ke checkpoint
    checkpoint_path = 'checkpoints/model_rasgan_50.pt'
    
    # Cek apakah file ada
    if not os.path.exists(checkpoint_path):
        print(f"❌ File {checkpoint_path} tidak ditemukan!")
        print("📝 Silakan sesuaikan path checkpoint di variabel 'checkpoint_path'")
        return
    
    # Inisialisasi extractor
    print("🚀 Memulai ekstraksi checkpoint...")
    extractor = CheckpointExtractor(checkpoint_path)
    
    # Tampilkan info checkpoint
    info = extractor.get_model_info()
    print("\n📋 Informasi Checkpoint:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Ekstraksi generator
    print("\n" + "="*60)
    extractor.extract_all_layers('generator', formats=['csv', 'binary'])
    
    # Ekstraksi discriminator
    print("\n" + "="*60)
    extractor.extract_all_layers('discriminator', formats=['csv', 'binary'])
    
    # Buat laporan arsitektur
    print("\n" + "="*60)
    extractor.create_architecture_report()
    
    # Export untuk Verilog/FPGA
    print("\n" + "="*60)
    print("\n🔧 Mengekspor format Verilog/FPGA...")
    extractor.export_for_verilog('generator')
    extractor.export_for_verilog('discriminator')
    
    print("\n" + "="*60)
    print("✅ SELESAI! Semua parameter berhasil diekstrak.")
    print(f"📁 Lokasi output: {extractor.output_dir}")
    print("\n📚 Format yang dihasilkan:")
    print("  1. CSV files - untuk analisis di Excel/Spreadsheet")
    print("  2. Binary files (.f32, .f16, .q15) - untuk loading langsung ke FPGA")
    print("  3. Hex files (.hex) - untuk Verilog $readmemh")
    print("  4. Verilog ROM modules (.v) - template siap pakai")
    print("  5. JSON metadata - informasi layer dan arsitektur")
    print("  6. Excel summary - ringkasan semua layer")


if __name__ == '__main__':
    main()