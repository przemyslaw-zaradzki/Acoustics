#!/usr/bin/env python3
"""
Eksperyment 1: Akustyczny Sonar ToF (Time-of-Flight) - Wersja dla Termux (Android)
Pomiar odległości za pomocą korelacji wzajemnej sygnału ćwierku (chirp).

Wymagania w Termux:
1. Pakiety Termux:
   pkg install python python-numpy python-scipy termux-api
2. Aplikacja pomocnicza na telefonie:
   Zainstalowana aplikacja "Termux:API" (z F-Droid) z przyznanymi uprawnieniami do mikrofonu i pamięci.
"""

import os
import sys
import tempfile
import argparse
import subprocess
import time
import shutil
import numpy as np
from scipy import signal
from scipy.io import wavfile

# Na Androidzie /tmp jest zablokowane przez SELinux.
# Używamy zmiennej $PREFIX/tmp lub standardowego katalogu tymczasowego Termux.
TMP_DIR = os.environ.get("TMPDIR", os.environ.get("PREFIX", "") + "/tmp" if os.environ.get("PREFIX") else tempfile.gettempdir())
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR, exist_ok=True)

TEMP_TX_WAV = os.path.join(TMP_DIR, "sonar_tx.wav")
TEMP_RX_RAW = os.path.join(TMP_DIR, "sonar_rx.m4a")
TEMP_RX_WAV = os.path.join(TMP_DIR, "sonar_rx.wav")


def check_termux_api():
    """Sprawdza, czy polecenia termux-api i ffmpeg są dostępne w systemie."""
    cmds = ["termux-microphone-record", "termux-media-player", "ffmpeg"]
    missing = [c for c in cmds if shutil.which(c) is None]
    if missing:
        print("\n" + "!" * 60)
        print("[BŁĄD] Brakujące polecenia systemowe:", ", ".join(missing))
        print("Aby je zainstalować, wykonaj w konsoli Termux:")
        print("    pkg install termux-api ffmpeg")
        print("\nUpewnij się również, że:")
        print(" 1. Zainstalowałeś aplikację 'Termux:API' z F-Droid.")
        print(" 2. Nadałeś jej uprawnienia do Mikrofonu:")
        print("    Ustawienia Androida -> Aplikacje -> Termux:API -> Uprawnienia -> Mikrofon (Zezwalaj)")
        print("!" * 60 + "\n")
        return False
    return True


def generate_chirp_wav(filename, f_start=2000, f_end=8000, duration=0.080, fs=44100, pre_silence=0.05, post_silence=0.25):
    """
    Tworzy plik WAV z ćwierkiem o zadanym czasie trwania i pasmie częstotliwości,
    otoczonym buforami ciszy.
    """
    n_chirp = int(fs * duration)
    t = np.linspace(0, duration, n_chirp, endpoint=False)
    chirp_raw = signal.chirp(t, f0=f_start, t1=duration, f1=f_end, method='linear')
    window = signal.windows.tukey(n_chirp, alpha=0.1)
    chirp = (chirp_raw * window).astype(np.float32)

    n_pre = int(fs * pre_silence)
    n_post = int(fs * post_silence)
    tx_signal = np.pad(chirp, (n_pre, n_post), mode='constant')

    # Konwersja na 16-bit PCM WAV
    audio_int16 = np.int16(tx_signal * 32767)
    wavfile.write(filename, fs, audio_int16)
    return chirp, fs, n_pre


def calculate_speed_of_sound(temperature_c=21.0):
    """Oblicza prędkość dźwięku w powietrzu w m/s."""
    return 331.3 * np.sqrt(1.0 + temperature_c / 273.15)


def run_termux_sonar(
    f_start=2000,
    f_end=8000,
    chirp_duration=0.080,
    temperature_c=21.0,
    calib_offset_m=0.0,
    two_way=False
):
    c = calculate_speed_of_sound(temperature_c)
    fs = 44100  # Domyślny stabilny próbnik w Android Termux

    # 1. Generowanie wzorcowego ćwierku i pliku TX
    pre_silence = 0.05
    post_silence = 0.35
    total_duration = pre_silence + chirp_duration + post_silence

    chirp, fs, pre_samples = generate_chirp_wav(
        TEMP_TX_WAV,
        f_start=f_start,
        f_end=f_end,
        duration=chirp_duration,
        fs=fs,
        pre_silence=pre_silence,
        post_silence=post_silence
    )

    # 2. Usuń stare pliki nagrania jeśli istnieją
    for f in [TEMP_RX_RAW, TEMP_RX_WAV]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    print(f"[*] Prędkość dźwięku (T={temperature_c:.1f}°C): c = {c:.2f} m/s")
    print(f"[*] Zakres ćwierku: {f_start} Hz - {f_end} Hz ({chirp_duration*1000:.0f} ms)")
    print(f"[*] Rozpoczynanie nagrywania mikrofonem ({total_duration:.2f} s)...")

    # Rozpocznij nagrywanie w tle przez termux-microphone-record (koder aac, częstotliwość 44100 Hz)
    rec_proc = subprocess.Popen(
        ["termux-microphone-record", "-f", TEMP_RX_RAW, "-e", "aac", "-r", "44100"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(0.15)  # Krótka pauza, by mikrofon zdążył wystartować

    # 3. Odtwórz sygnał nadawczy
    print("[*] Emisja ćwierku przez głośnik...")
    subprocess.run(["termux-media-player", "play", TEMP_TX_WAV], check=False)

    # Czekaj na zakończenie nagrywania
    time.sleep(total_duration + 0.3)

    # Zatrzymaj nagrywanie
    subprocess.run(["termux-microphone-record", "-q"], check=False)
    try:
        rec_proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        rec_proc.kill()

    # Daj Androidowi chwilę na flush i domknięcie pliku m4a (zapis kontenera moov)
    time.sleep(0.3)

    # 4. Sprawdzenie i konwersja nagranego pliku
    # Czekaj max 2 sekundy aż plik pojawi się na dysku i nie będzie pusty
    for _ in range(20):
        if os.path.exists(TEMP_RX_RAW) and os.path.getsize(TEMP_RX_RAW) > 500:
            break
        time.sleep(0.1)

    if not os.path.exists(TEMP_RX_RAW) or os.path.getsize(TEMP_RX_RAW) == 0:
        print("\n[BŁĄD] Plik nagrania nie powstał lub jest pusty.")
        print("Upewnij się, że aplikacja Termux:API posiada uprawnienie do Mikrofonu:")
        print("Ustawienia telefonu -> Aplikacje -> Termux:API -> Uprawnienia -> Mikrofon (Zezwalaj).")
        return

    # Konwersja AAC/M4A -> czysty nieskompresowany WAV (PCM 16-bit) przez ffmpeg
    conv = subprocess.run(
        ["ffmpeg", "-y", "-i", TEMP_RX_RAW, "-ar", str(fs), "-ac", "1", TEMP_RX_WAV],
        capture_output=True,
        text=True
    )
    if conv.returncode != 0 or not os.path.exists(TEMP_RX_WAV):
        print("\n[BŁĄD] Konwersja nagrania do formatu WAV za pomocą ffmpeg nie powiodła się.")
        if conv.stderr:
            err_lines = [line for line in conv.stderr.splitlines() if "Error" in line or "Invalid" in line]
            if err_lines:
                print("Szczegóły:", "\n".join(err_lines))
        return

    try:
        sample_rate, rx_raw = wavfile.read(TEMP_RX_WAV)
    except Exception as e:
        print(f"[BŁĄD] Błąd odczytu pliku nagrania WAV: {e}")
        return

    # Sprowadzenie do mono i float32
    if rx_raw.ndim > 1:
        rx_raw = rx_raw.mean(axis=1)
    rx_signal = rx_raw.astype(np.float32) / 32768.0

    # Resampluj wzorzec jeśli system nagrał z inną częstotliwością niż 44.1 kHz
    if sample_rate != fs:
        num_samples = int(len(chirp) * sample_rate / fs)
        chirp_resampled = signal.resample(chirp, num_samples)
        pre_samples = int(pre_samples * sample_rate / fs)
    else:
        chirp_resampled = chirp

    # 5. Filtracja pasmowoprzepustowa
    nyquist = 0.5 * sample_rate
    low_cut = max((f_start * 0.8) / nyquist, 0.01)
    high_cut = min((f_end * 1.2) / nyquist, 0.99)
    b, a = signal.butter(4, [low_cut, high_cut], btype='band')
    rx_filtered = signal.filtfilt(b, a, rx_signal)

    # 6. Korelacja wzajemna (filtr dopasowany)
    correlation = signal.correlate(rx_filtered, chirp_resampled, mode='valid', method='fft')
    corr_envelope = np.abs(signal.hilbert(correlation))
    corr_envelope /= (np.max(corr_envelope) + 1e-12)

    time_flight = (np.arange(len(corr_envelope)) - pre_samples) / sample_rate

    # Wykrycie piku
    valid_mask = time_flight >= 0
    t_valid = time_flight[valid_mask]
    corr_valid = corr_envelope[valid_mask]

    peaks, properties = signal.find_peaks(corr_valid, height=0.3, distance=int(sample_rate * 0.002))
    
    if len(peaks) == 0:
        best_peak_idx = np.argmax(corr_valid)
    else:
        best_peak_idx = peaks[np.argmax(properties['peak_heights'])]

    tof_sec = t_valid[best_peak_idx]
    divisor = 2.0 if two_way else 1.0
    distance_raw_m = (tof_sec * c) / divisor
    distance_calibrated_m = max(0.0, distance_raw_m - calib_offset_m)

    mode_str = "Bezpośredni 1-way (Głośnik -> Mikrofon)" if not two_way else "Echo 2-way"
    print("\n" + "="*50)
    print(f"  WYNIK POMIARU (Android Termux: {mode_str})")
    print("="*50)
    print(f"  Czas przelotu (ToF):  {tof_sec * 1000.0:.2f} ms")
    print(f"  Zmierzona odległość:  {distance_calibrated_m * 100.0:.2f} cm ({distance_calibrated_m:.3f} m)")
    if calib_offset_m > 0:
        print(f"  (Zastosowany offset: -{calib_offset_m * 100.0:.2f} cm)")
    print("="*50 + "\n")

    # Wypisanie alternatywnych pików
    if len(peaks) > 1:
        print("Wykryte echa / odbicia:")
        for i, p in enumerate(peaks):
            p_dist = (t_valid[p] * c) / divisor - calib_offset_m
            marker = " <-- GŁÓWNY SYGNAŁ" if p == best_peak_idx else ""
            print(f"  Pik #{i+1}: t = {t_valid[p]*1000.0:6.1f} ms | d = {p_dist*100.0:6.1f} cm | korelacja = {corr_valid[p]:.2f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Akustyczny Sonar ToF dla Androida (Termux)")
    parser.add_argument("--fmin", type=float, default=2000, help="Początek ćwierku Hz (domyślnie 2000)")
    parser.add_argument("--fmax", type=float, default=8000, help="Koniec ćwierku Hz (domyślnie 8000)")
    parser.add_argument("--duration", type=float, default=0.080, help="Czas trwania w sekundach (domyślnie 0.08s)")
    parser.add_argument("--temp", type=float, default=21.0, help="Temperatura w °C (domyślnie 21.0)")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset kalibracyjny w metrach")
    parser.add_argument("--echo", action="store_true", help="Tryb 2-way echa (odbicia od ściany)")
    args = parser.parse_args()

    if not check_termux_api():
        sys.exit(1)
    run_termux_sonar(
        f_start=args.fmin,
        f_end=args.fmax,
        chirp_duration=args.duration,
        temperature_c=args.temp,
        calib_offset_m=args.offset,
        two_way=args.echo
    )


if __name__ == "__main__":
    main()

