
# WaveLLDM: Desain dan Pengembangan Model Difusi Laten Ringan dalam Peningkatan Kualitas dan Restorasi Suara Ucapan
Kevin Putra Santoso (2025)

Wave Lightweight Latent Diffusion Models (WaveLLDM) adalah model difusi laten berbasis deep learning yang dirancang untuk melakukan proses denoising audio pada perangkat tepi. WaveLLDM bertujuan meningkatkan kualitas dan restorasi suara ucapan dengan efisiensi komputasi yang lebih baik dibandingkan model difusi konvensional.

![WaveLLDM Architecture](https://github.com/echelon2718/WaveLLDM/blob/main_new/assets/WaveLLDM_Arch.png)

## 📌 **Latar Belakang**
Model berbasis difusi telah menunjukkan keunggulan dalam sintesis audio, dengan stabilitas pelatihan yang lebih baik dibandingkan Generative Adversarial Networks (GAN) dan model autoregresif. Namun, tantangan utama model ini adalah tingginya kebutuhan daya komputasi. WaveLLDM dikembangkan sebagai solusi dengan arsitektur yang lebih ringan, memungkinkan implementasi pada perangkat tepi seperti Android dan aplikasi web.

## ⚙ **Komponen Neural Audio Codec WaveLLDM**
WaveLLDM terdiri dari tiga komponen utama dalam generator:
1. **ConvNeXt Encoder**: Memproses spektogram mel audio menjadi representasi laten $z$ dengan dimensi laten **384 × L**.
2. **Grouped Residual Finite Scalar Quantization (GFSQ)**: Melakukan kuantisasi pada $\vec{z}$ menjadi $\vec{z_q}$ menggunakan GFSQ. WaveLLDM menggunakan **4 codebook**, dengan level per codebook: **[8, 8, 8, 6, 5]**.
3. **HiFi-GAN Decoder**: Mengembalikan $\vec{z_q}$ menjadi audio asal.
![FFGAN Components](https://github.com/echelon2718/WaveLLDM/blob/main_new/assets/FireflyGAN_Components.png)

Selain itu, Neural Audio Codec WaveLLDM dilengkapi dengan **HiFi-GAN Discriminator**: Multi-Period Discriminator (MPD) dan Multi-Scale Discriminator (MSD).

## 🎯 **Loss Functions**
Model ini dilatih menggunakan kombinasi dari empat fungsi loss:
1. **Adversarial Loss**
2. **Feature Matching Loss** (Lambda = **2**)
3. **Multiscale Mel Spectrogram Loss** (Lambda = **65**)
4. **Multiscale STFT Spectral Loss** (Lambda = **20**)

## 🔨 **Arsitektur Diffusion**
Selanjutnya, kami mengusulkan **Rotary U-Net**, sebuah model **U-Net** yang bekerja di **latent space** untuk melakukan difusi laten. Arsitektur U-Net ini akan didukung dengan **Parallel 1D Residual Convolutional Block** dan **Rotary Attention Mechanism**, yang membantu dalam memahami konteks sekuensial audio.

![WaveLLDM Architecture](https://github.com/echelon2718/WaveLLDM/blob/main_new/assets/Rotary_UNET.png)

WaveLLDM dilatih menggunakan prinsip **Denoising Diffusion Probabilistic Models (DDPM)** dalam **ruang laten**, dengan proses difusi maju dan balik sebagai berikut. 

### 1. 🔄 Proses Difusi Maju (Forward Diffusion)
Model menambahkan noise secara bertahap ke representasi laten $z_0$ hingga menjadi noise murni $z_T$. Proses ini dinyatakan dengan:

$$
z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

dengan:
- $\alpha_t = 1 - \beta_t$  
- $\bar{\alpha}\_t = \prod_{s=1}^t \alpha_s$  
- $\beta_t$ ditentukan melalui **cosine beta schedule** (Nichol & Dhariwal, 2021).

### 2. 🔁 Proses Reverse (Sampling)
Model mempelajari distribusi balik $p_\theta(z_{t-1} \mid z_t)$, yang diasumsikan Gaussian:

$$
p_\theta(z_{t-1} \mid z_t) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t), \Sigma_\theta(z_t, t) I)
$$

Dengan parameterisasi:
  
$$
\mu_\theta(z_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(z_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}\_t}} \epsilon_\theta(z_t, t) \right)
$$  

$$
\Sigma_\theta(z_t, t) = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t
$$

### 3. 🧠 Tujuan Pelatihan

Alih-alih merekonstruksi $z_0$, model cukup memprediksi noise ($\epsilon$) yang ditambahkan. Loss yang digunakan:  

$$
\mathcal{L}\_{\text{simple}} = \mathbb{E}\_{z_0, \epsilon, t} \left\[ \left\| \epsilon - \epsilon_\theta \left(\sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right\|^2 \right\]
$$  

Loss ini dikenal sebagai **denoising score matching loss**, dan cukup untuk mempelajari proses reverse difusi.

## 📂 **Dataset**
- **Training:** Voicebank+DEMAND dan dataset terkait lainnya.
- **Evaluation:** LJSpeech dengan metrik **Mean Opinion Score (MOS)**.

Struktur Filesystem Dataset:
```
voicebank_demand_56spk/
├── clean_speech_audios/
│   ├── train/
│   │   ├── p234_001.wav
│   │   ├── p234_002.wav
│   │   └── ...
│   └── test/
│       ├── p232_001.wav
│       ├── p232_002.wav
│       └── ...
├── noisy_speech_audios/
    ├── train/
    │   ├── p234_001.wav
    │   ├── p234_002.wav
    │   └── ...
    └── test/
        ├── p232_001.wav
        ├── p232_002.wav
        └── ...
```

## 🚀 **Instalasi dan Penggunaan**
### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Training Model**
Gunakan skrip `train_vctk.py` dengan parameter berikut (sesuaikan jika diperlukan) untuk melatih model tahap 1 (1st stage training):
```bash
python train_vctk.py \
  --clean_data_path ./data/vctk/voicebank_demand_56spk/clean_speech_audios/train/ \
  --noisy_data_path ./data/vctk/voicebank_demand_56spk/noisy_speech_audios/train/ \
  --add_random_cutting True \
  --batch_size 16 \
  --lr 2e-5 \
  --use_scheduler True \
  --epochs 100 \
  --num_workers 16
```

Gunakan skrip `train_wavelldm.py` dengan parameter berikut untuk melatih model tahap 2 (2nd stage training):
```bash
python train_wavelldm.py \
  --clean_data_path ./data/vctk/voicebank_demand_56spk/clean_speech_audios/train/ \
  --noisy_data_path ./data/vctk/voicebank_demand_56spk/noisy_speech_audios/train/ \
  --add_random_cutting True \
  --max_cuts 7 \
  --cut_duration 0.45 \ # inpainting 50 - 450ms
  --batch_size 64 \
  --val_batch_size 16 \
  --epochs 400 \
  --lr 3e-5 \
  --num_workers 16 \
  --pretrained_codec_path ./pretrained_models/generator_step_142465.pth
```

## ✅ **Progress Saat Ini**
✔ Perancangan Neural Audio Codec  
✔ Perancangan Arsitektur U-Net  
✔ Training Cycle  
✔ Loss Functions, Dataset, Data Loader, dan modul pendukung lainnya  
✔ Melatih U-Net untuk DDPM 
✔ Pengujian end-to-end  
✔ Implementasi kode inferensi  
✔ Implementasi UI dengan **Gradio**  

## 📌 **Rencana Implementasi**
Setelah model difusi selesai, WaveLLDM akan diimplementasikan dalam:
- **Aplikasi Web** untuk demonstrasi real-time.
- **Perangkat Tepi (Android)** untuk denoising suara langsung dari perangkat.

## 📄 **Lisensi**
Properti intelektual ini dilindungi oleh lisensi CC-BY-NC. Baca dokumen LICENSE-CC-BY-NC untuk keterangan lebih lanjut
