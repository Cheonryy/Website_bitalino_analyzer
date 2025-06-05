import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.signal import butter, filtfilt, welch
from scipy.stats import linregress
import io # Diperlukan untuk menangani file yang diunggah

# === 1. Fungsi Bantu untuk Pemrosesan Sinyal EMG ===
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = fs / 2
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, notch_freq, fs, order=2):
    nyq = fs / 2
    # Rentang filter notch sedikit diperluas untuk lebih efektif
    low = (notch_freq - 1) / nyq
    high = (notch_freq + 1) / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, signal)

def convert_to_mv(raw, n_bits, vcc=3.3, gain=1000):
    # Rumus konversi untuk Bitalino
    # Sesuaikan vcc dan gain jika perangkat Bitalino Anda berbeda
    return ((raw / (2**n_bits)) - 0.5) * vcc * 1000 / gain

# === 2. Fungsi Utama Pemrosesan EMG ===
def process_emg_file(uploaded_file, baseline_duration_sec=5):
    # Membaca seluruh konten file sebagai string
    string_data = uploaded_file.getvalue().decode("utf-8")
    
    # === 1. Load file dan ekstrak header ===
    fs = 1000 # Default
    resolution_bits = 10 # Default
    emg_column_index = 5 # Default
    
    try:
        # Membaca baris kedua untuk header JSON
        lines = string_data.splitlines()
        if len(lines) > 1 and lines[1].startswith('#'):
            header_json_str = lines[1][1:].strip()
            header_json = json.loads(header_json_str)
            device_id = list(header_json.keys())[0]
            info = header_json[device_id]
            
            fs = info.get("sampling rate", fs)
            columns = info.get("column", [])
            
            if "A1" in columns:
                emg_column_index = columns.index("A1")
                # Pastikan indeks resolusi valid
                if emg_column_index < len(info.get("resolution", [])):
                    resolution_bits = info.get("resolution", [])[emg_column_index]
                else:
                    st.warning(f"Resolusi untuk kolom A1 tidak ditemukan di header. Menggunakan nilai default: {resolution_bits} bit.")
            else:
                st.error("Kolom 'A1' (EMG) tidak ditemukan di header file Anda. Harap pastikan file adalah rekaman EMG Bitalino yang valid.")
                return None, None, None, None, None, None, None, None, None
        else:
            st.warning("Header file tidak ditemukan atau tidak dalam format yang diharapkan. Menggunakan nilai default.")
            
    except json.JSONDecodeError as e:
        st.error(f"Gagal membaca header JSON. Format mungkin tidak valid: {e}. Menggunakan nilai default.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengekstrak header: {e}. Menggunakan nilai default.")

    win_size = fs // 4 # 0.25 detik
    step = fs // 8     # Overlap 50%    
    
    # --- TAMBAHKAN BARIS INI UNTUK MENYIMPAN KE SESSION STATE ---
    st.session_state.fs = fs
    st.session_state.resolution_bits = resolution_bits
    st.session_state.emg_column_index = emg_column_index
    st.session_state.win_size = win_size / fs # Simpan juga ukuran window dalam detik

    # === 2. Load data dan buat kolom waktu ===
    # Menggunakan io.StringIO untuk membaca string_data seolah-olah itu file
    df = pd.read_csv(io.StringIO(string_data), sep='\s+', comment="#", header=None, usecols=[0, emg_column_index])
    df.columns = ['nSeq', 'emg_raw']
    df['time'] = np.arange(len(df)) / fs

    # === 4. Pre-processing ===
    df['emg_mv'] = convert_to_mv(df['emg_raw'], resolution_bits)
    # Penghapusan offset DC dilakukan setelah konversi ke mV
    df['emg_processed'] = df['emg_mv'] - df['emg_mv'].mean()
    df['emg_processed'] = notch_filter(df['emg_processed'], 50, fs)
    df['emg_processed'] = bandpass_filter(df['emg_processed'], 20, 450, fs)
    df['emg_rectified'] = np.abs(df['emg_processed'])

    # === 5. Fitur (RMS dan MNF per window yang lebih kecil) ===
    rms_vals, mnf_vals, t_vals = [], [], []

    for i in range(0, len(df) - win_size + 1, step):
        segment = df['emg_rectified'].iloc[i:i + win_size].values
        t = df['time'].iloc[i + win_size // 2]

        rms = np.sqrt(np.mean(segment**2))
        # nperseg disesuaikan agar tidak lebih besar dari window size
        f, Pxx = welch(segment, fs=fs, nperseg=min(128, win_size))
        mnf = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0

        rms_vals.append(rms)
        mnf_vals.append(mnf)
        t_vals.append(t)
    
    # --- Perhitungan Baseline ---
    rms_baseline, mnf_baseline = 0, 0
    num_windows_for_baseline = min(int(baseline_duration_sec / (step / fs)), len(rms_vals))

    if num_windows_for_baseline == 0:
        st.warning("Tidak cukup data untuk menghitung baseline dengan durasi yang ditentukan. Baseline diset 0.")
    else:
        rms_baseline = np.mean(rms_vals[:num_windows_for_baseline])
        mnf_baseline = np.mean(mnf_vals[:num_windows_for_baseline])

    # --- Analisis Tren (Slope) ---
    slope_rms, p_value_rms = None, None
    slope_mnf, p_value_mnf = None, None
    if len(t_vals) > 1: # Memastikan ada cukup data untuk regresi linear
        slope_rms, _, _, p_value_rms, _ = linregress(t_vals, rms_vals)
        slope_mnf, _, _, p_value_mnf, _ = linregress(t_vals, mnf_vals)


    return df, t_vals, rms_vals, mnf_vals, rms_baseline, mnf_baseline, slope_rms, p_value_rms, slope_mnf, p_value_mnf

# === 3. Aplikasi Streamlit ===
st.set_page_config(layout="wide") # Mengatur layout agar lebih lebar
st.title("EMG Bitalino Analyzer")
st.markdown("Unggah file `.txt` OpenSignals Anda untuk melihat analisis RMS dan Mean Frequency (MNF) sinyal EMG.")

uploaded_file = st.file_uploader("Pilih file .txt EMG Bitalino", type="txt")

if uploaded_file is not None:
    # Menggunakan st.spinner untuk menampilkan indikator loading
    with st.spinner("Memproses data EMG..."):
        df_raw, t_vals, rms_vals, mnf_vals, rms_baseline, mnf_baseline, slope_rms, p_value_rms, slope_mnf, p_value_mnf = process_emg_file(uploaded_file)

    if df_raw is None: # Jika ada error pada proses_emg_file
        st.error("Gagal memproses file. Pastikan format file sudah benar.")
    else:
        st.success("File berhasil diproses!")
        
        # === Tampilkan Informasi Header & Baseline ===
        st.subheader("Informasi Terdeteksi dari Header:")
        st.write(f"**Sampling Rate (fs):** {st.session_state.get('fs', 1000)} Hz")
        st.write(f"**Resolusi (bits):** {st.session_state.get('resolution_bits', 10)} bit")
        st.write(f"**Indeks Kolom EMG (A1):** {st.session_state.get('emg_column_index', 5)}")
        
        st.subheader("Nilai Baseline (5 detik pertama):")
        st.write(f"**RMS Baseline:** {rms_baseline:.4f} mV")
        st.write(f"**MNF Baseline:** {mnf_baseline:.2f} Hz")

        # === Tampilkan Grafik ===
        st.subheader("Grafik Analisis Sinyal EMG")

        fig = plt.figure(figsize=(16, 12))

        # Subplot 1: Sinyal Raw
        plt.subplot(4, 1, 1)
        plt.plot(df_raw['time'], df_raw['emg_mv'], label='Raw EMG (mV)', color='gray', alpha=0.7)
        plt.title('Sinyal EMG Mentah (dalam mV)')
        plt.ylabel('Tegangan (mV)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Subplot 2: Sinyal Setelah Preprocessing
        plt.subplot(4, 1, 2)
        plt.plot(df_raw['time'], df_raw['emg_rectified'], label='Filtered & Rectified', color='purple', alpha=0.8)
        plt.title('Sinyal EMG setelah Preprocessing (20-450Hz Bandpass, 50Hz Notch, Rectified)')
        plt.ylabel('mV')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Subplot 3: RMS
        plt.subplot(4, 1, 3)
        plt.plot(t_vals, rms_vals, label='RMS', color='blue', linewidth=2)
        plt.axhline(y=rms_baseline, color='green', linestyle='--', label=f'RMS Baseline ({rms_baseline:.4f} mV)')
        plt.title(f'Root Mean Square (RMS) - Window: {st.session_state.get("win_size", 0.25):.3f}s')
        plt.xlabel('Waktu (s)')
        plt.ylabel('RMS (mV)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if len(rms_vals) > 0:
            rms_min, rms_max = min(rms_vals), max(rms_vals)
            plt.ylim(rms_min * 0.9, rms_max * 1.1)

        # Subplot 4: MNF
        plt.subplot(4, 1, 4)
        plt.plot(t_vals, mnf_vals, label='MNF', color='red', linewidth=2)
        plt.axhline(y=mnf_baseline, color='green', linestyle='--', label=f'MNF Baseline ({mnf_baseline:.2f} Hz)')
        plt.title('Mean Frequency (MNF)')
        plt.xlabel('Waktu (s)')
        plt.ylabel('Frekuensi (Hz)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if len(mnf_vals) > 0:
            mnf_filtered = [x for x in mnf_vals if x > 0]
            if mnf_filtered:
                mnf_min, mnf_max = min(mnf_filtered), max(mnf_filtered)
                plt.ylim(max(0, mnf_min * 0.9), mnf_max * 1.1)

        plt.tight_layout()
        st.pyplot(fig) # Menampilkan figur Matplotlib di Streamlit

        # === Tampilkan Analisis Kelelahan ===
        st.subheader("Analisis Kelelahan Otot")

        if len(t_vals) > 0 and rms_baseline != 0 and mnf_baseline != 0:
            final_rms = rms_vals[-1]
            final_mnf = mnf_vals[-1]

            rms_change_percent = ((final_rms - rms_baseline) / rms_baseline) * 100 if rms_baseline != 0 else float('inf') * np.sign(final_rms - rms_baseline)
            mnf_change_percent = ((final_mnf - mnf_baseline) / mnf_baseline) * 100 if mnf_baseline != 0 else float('inf') * np.sign(final_mnf - mnf_baseline)

            st.write(f"**Perubahan RMS (akhir rekaman vs baseline):** {rms_change_percent:.2f}%")
            st.write(f"**Perubahan MNF (akhir rekaman vs baseline):** {mnf_change_percent:.2f}%")

            st.markdown("---")
            st.write("**Kesimpulan Awal (Berdasarkan Perubahan Persentase):**")
            FATIGUE_THRESHOLD_MNF = st.slider("Ambang Batas Penurunan MNF untuk Kelelahan (%)", min_value=-50.0, max_value=0.0, value=-10.0, step=1.0)
            FATIGUE_THRESHOLD_RMS = st.slider("Ambang Batas Penurunan RMS untuk Kelelahan (%)", min_value=-50.0, max_value=0.0, value=-10.0, step=1.0)

            is_fatigue_mnf = mnf_change_percent <= FATIGUE_THRESHOLD_MNF
            is_fatigue_rms = rms_change_percent <= FATIGUE_THRESHOLD_RMS

            if is_fatigue_mnf and is_fatigue_rms:
                st.markdown("<h4 style='color:red;'>STATUS: Otot terdeteksi FATIGUE (MNF dan RMS keduanya menurun signifikan).</h4>", unsafe_allow_html=True)
            elif is_fatigue_mnf:
                st.markdown("<h4 style='color:red;'>STATUS: Otot terdeteksi FATIGUE (MNF menurun signifikan, indikator kuat).</h4>", unsafe_allow_html=True)
            elif is_fatigue_rms:
                st.markdown("<h4 style='color:orange;'>STATUS: Otot terdeteksi FATIGUE (RMS menurun signifikan, perlu dikonfirmasi dengan MNF).</h4>", unsafe_allow_html=True)
            else:
                st.markdown("<h4 style='color:green;'>STATUS: Otot TIDAK terdeteksi FATIGUE (berdasarkan kriteria persentase perubahan).</h4>", unsafe_allow_html=True)
            
            st.markdown("---")

            st.write("**Analisis Tren Linier (Seluruh Durasi Rekaman):**")
            if slope_mnf is not None and slope_rms is not None:
                st.write(f"**Slope RMS:** {slope_rms:.4f} mV/detik (p-value: {p_value_rms:.4f})")
                st.write(f"**Slope MNF:** {slope_mnf:.4f} Hz/detik (p-value: {p_value_mnf:.4f})")
                
                SIGNIFICANCE_ALPHA = st.slider("Tingkat Signifikansi (p-value)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

                is_fatigue_trend_mnf = slope_mnf < 0 and p_value_mnf < SIGNIFICANCE_ALPHA
                is_fatigue_trend_rms = slope_rms < 0 and p_value_rms < SIGNIFICANCE_ALPHA

                if is_fatigue_trend_mnf and is_fatigue_trend_rms:
                    st.markdown("<h4 style='color:red;'>STATUS TREN: Otot menunjukkan tren kelelahan (MNF dan RMS menurun secara signifikan secara statistik).</h4>", unsafe_allow_html=True)
                elif is_fatigue_trend_mnf:
                    st.markdown("<h4 style='color:red;'>STATUS TREN: Otot menunjukkan tren kelelahan (MNF menurun signifikan secara statistik).</h4>", unsafe_allow_html=True)
                elif is_fatigue_trend_rms:
                    st.markdown("<h4 style='color:orange;'>STATUS TREN: Otot menunjukkan tren kelelahan (RMS menurun signifikan secara statistik, perlu dikonfirmasi dengan MNF).</h4>", unsafe_allow_html=True)
                else:
                    st.markdown("<h4 style='color:green;'>STATUS TREN: Otot tidak menunjukkan tren kelelahan yang signifikan secara statistik.</h4>", unsafe_allow_html=True)
            else:
                st.warning("Tidak cukup data untuk melakukan analisis tren linier.")
            
            st.markdown("---")
            st.info("Catatan: Interpretasi kelelahan otot harus selalu mempertimbangkan konteks eksperimen dan karakteristik individu.")

else:
    st.info("Unggah file .txt EMG Bitalino untuk memulai analisis.")
