#!/usr/bin/env python3
"""
Eksperyment 1: Akustyczny Sonar ToF (Time-of-Flight) - Wersja dla Termux (Android)
Pomiar odległości za pomocą korelacji wzajemnej sygnału ćwierku (chirp).

Wymagania w Termux:
1. Pakiety Termux:
   pkg install python python-numpy python-scipy termux-api ffmpeg
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


def generate_chirp_wav(
    filename,
    f_start=2000,
    f_end=8000,
    duration=0.080,
    fs=44100,
    pre_silence=0.05,
    post_silence=0.25,
    warmup_duration=0.5,
    warmup_freq=440.0,
    warmup_gain=0.02
):
    """
    Tworzy plik WAV z ćwierkiem o zadanym czasie trwania i pasmie częstotliwości,
    poprzedzonym pilotem / tonem wybudzającym (warmup tone) dla głośników Bluetooth
    oraz buforami ciszy.
    """
    # 1. Warm-up tone (pilot wybudzający DSP wzmacniacza Bluetooth)
    if warmup_duration > 0:
        n_warmup = int(fs * warmup_duration)
        t_w = np.linspace(0, warmup_duration, n_warmup, endpoint=False)
        w_window = signal.windows.tukey(n_warmup, alpha=0.1)
        warmup_signal = (np.sin(2 * np.pi * warmup_freq * t_w) * warmup_gain * w_window).astype(np.float32)
    else:
        warmup_signal = np.array([], dtype=np.float32)

    # 2. Główny sygnał pomiarowy (chirp)
    n_chirp = int(fs * duration)
    t = np.linspace(0, duration, n_chirp, endpoint=False)
    chirp_raw = signal.chirp(t, f0=f_start, t1=duration, f1=f_end, method='linear')
    window = signal.windows.tukey(n_chirp, alpha=0.1)
    chirp = (chirp_raw * window).astype(np.float32)

    n_pre = int(fs * pre_silence)
    n_post = int(fs * post_silence)

    tx_parts = []
    if len(warmup_signal) > 0:
        tx_parts.append(warmup_signal)
    if n_pre > 0:
        tx_parts.append(np.zeros(n_pre, dtype=np.float32))

    pre_samples = sum(len(p) for p in tx_parts)
    tx_parts.append(chirp)
    if n_post > 0:
        tx_parts.append(np.zeros(n_post, dtype=np.float32))

    tx_signal = np.concatenate(tx_parts)

    audio_int16 = np.int16(np.clip(tx_signal, -1.0, 1.0) * 32767)
    wavfile.write(filename, fs, audio_int16)
    return chirp, fs, pre_samples


def calculate_speed_of_sound(temperature_c=21.0):
    """Oblicza prędkość dźwięku w powietrzu w m/s."""
    return 331.3 * np.sqrt(1.0 + temperature_c / 273.15)


def run_termux_sonar(
    f_start=2000,
    f_end=8000,
    chirp_duration=0.080,
    temperature_c=21.0,
    calib_offset_m=0.0,
    two_way=False,
    warmup_duration=0.5,
    debug=False,
    save_rx=None
):
    c = calculate_speed_of_sound(temperature_c)
    fs = 44100  # Domyślny stabilny próbnik w Android Termux
    t_start = time.perf_counter()
    debug_logs = []

    def dlog(msg):
        entry = f"[DEBUG +{(time.perf_counter()-t_start)*1000.0:6.1f} ms] {msg}"
        debug_logs.append(entry)
        if debug:
            print(entry)

    dlog("Rozpoczęcie procedury pomiarowej.")

    # 1. Generowanie wzorcowego ćwierku i pliku TX
    pre_silence = 0.05
    post_silence = 0.45
    total_duration = warmup_duration + pre_silence + chirp_duration + post_silence

    t_gen = time.perf_counter()
    chirp, fs, pre_samples = generate_chirp_wav(
        TEMP_TX_WAV,
        f_start=f_start,
        f_end=f_end,
        duration=chirp_duration,
        fs=fs,
        pre_silence=pre_silence,
        post_silence=post_silence,
        warmup_duration=warmup_duration
    )
    dlog(f"Wygenerowano TX WAV w {(time.perf_counter()-t_gen)*1000.0:.1f} ms. pre_samples={pre_samples}")

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

    t_rec = time.perf_counter()
    rec_proc = subprocess.Popen(
        ["termux-microphone-record", "-f", TEMP_RX_RAW, "-e", "aac", "-r", "44100"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    dlog(f"Uruchomiono nagrywanie PID={rec_proc.pid} w {(time.perf_counter()-t_rec)*1000.0:.1f} ms")

    time.sleep(0.15)  # Krótka pauza, by mikrofon zdążył wystartować

    # 3. Odtwórz sygnał nadawczy
    print("[*] Emisja ćwierku przez głośnik...")
    t_play = time.perf_counter()
    res_play = subprocess.run(["termux-media-player", "play", TEMP_TX_WAV], capture_output=True, text=True, check=False)
    dlog(f"termux-media-player play wywołany w {(time.perf_counter()-t_play)*1000.0:.1f} ms (exit={res_play.returncode})")

    # Czekaj na zakończenie nagrywania
    time.sleep(total_duration + 0.3)

    # Zatrzymaj nagrywanie
    dlog("Zatrzymywanie nagrywania mikrofonem...")
    t_stop = time.perf_counter()
    subprocess.run(["termux-microphone-record", "-q"], check=False)
    try:
        rec_proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        rec_proc.kill()
    dlog(f"Nagrywanie zatrzymane w {(time.perf_counter()-t_stop)*1000.0:.1f} ms")

    time.sleep(0.3)

    # 4. Sprawdzenie i konwersja nagranego pliku
    wait_start = time.perf_counter()
    for _ in range(25):
        if os.path.exists(TEMP_RX_RAW) and os.path.getsize(TEMP_RX_RAW) > 500:
            break
        time.sleep(0.1)
    raw_size = os.path.getsize(TEMP_RX_RAW) if os.path.exists(TEMP_RX_RAW) else 0
    dlog(f"Oczekiwanie na plik m4a: {(time.perf_counter()-wait_start)*1000.0:.1f} ms, rozmiar: {raw_size} bajtów")

    if not os.path.exists(TEMP_RX_RAW) or raw_size == 0:
        print("\n[BŁĄD] Plik nagrania nie powstał lub jest pusty.")
        print("Upewnij się, że aplikacja Termux:API posiada uprawnienie do Mikrofonu:")
        print("Ustawienia telefonu -> Aplikacje -> Termux:API -> Uprawnienia -> Mikrofon (Zezwalaj).")
        return

    t_conv = time.perf_counter()
    conv = subprocess.run(
        ["ffmpeg", "-y", "-i", TEMP_RX_RAW, "-ar", str(fs), "-ac", "1", TEMP_RX_WAV],
        capture_output=True,
        text=True
    )
    dlog(f"Konwersja ffmpeg do WAV w {(time.perf_counter()-t_conv)*1000.0:.1f} ms")

    if conv.returncode != 0 or not os.path.exists(TEMP_RX_WAV):
        print("\n[BŁĄD] Konwersja nagrania do formatu WAV za pomocą ffmpeg nie powiodła się.")
        if conv.stderr:
            err_lines = [line for line in conv.stderr.splitlines() if "Error" in line or "Invalid" in line]
            if err_lines:
                print("Szczegóły:\n" + "\n".join(err_lines))
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

    rx_len_s = len(rx_signal) / sample_rate
    rx_peak = np.max(np.abs(rx_signal))
    rx_rms = np.sqrt(np.mean(rx_signal**2))
    dlog(f"Parametry nagrania: długość={rx_len_s:.3f} s ({len(rx_signal)} próbek), pik_amp={rx_peak:.3f}, RMS={rx_rms:.4f}")

    # Resampluj wzorzec jeśli system nagrał z inną częstotliwością niż 44.1 kHz
    if sample_rate != fs:
        num_samples = int(len(chirp) * sample_rate / fs)
        chirp_resampled = signal.resample(chirp, num_samples)
        pre_samples = int(pre_samples * sample_rate / fs)
    else:
        chirp_resampled = chirp

    # 5. Filtracja pasmowoprzepustowa
    t_filt = time.perf_counter()
    nyquist = 0.5 * sample_rate
    low_cut = max((f_start * 0.8) / nyquist, 0.01)
    high_cut = min((f_end * 1.2) / nyquist, 0.99)
    b, a = signal.butter(4, [low_cut, high_cut], btype='band')
    rx_filtered = signal.filtfilt(b, a, rx_signal)
    dlog(f"Filtracja Butterwotha zakończona w {(time.perf_counter()-t_filt)*1000.0:.1f} ms")

    # 6. Korelacja wzajemna (filtr dopasowany)
    t_corr = time.perf_counter()
    correlation = signal.correlate(rx_filtered, chirp_resampled, mode='valid', method='fft')
    corr_envelope = np.abs(signal.hilbert(correlation))
    max_raw_corr = np.max(corr_envelope)
    corr_envelope /= (max_raw_corr + 1e-12)
    dlog(f"Korelacja FFT obliczona w {(time.perf_counter()-t_corr)*1000.0:.1f} ms. Max surowa wartość: {max_raw_corr:.2f}")

    time_flight = (np.arange(len(corr_envelope)) - pre_samples) / sample_rate

    # Wykrycie piku
    valid_mask = time_flight >= 0
    t_valid = time_flight[valid_mask]
    corr_valid = corr_envelope[valid_mask]

    peaks, properties = signal.find_peaks(corr_valid, height=0.25, distance=int(sample_rate * 0.002))

    if len(peaks) == 0:
        best_peak_idx = int(np.argmax(corr_valid))
        best_height = corr_valid[best_peak_idx]
        dlog(f"Brak wyraźnych pików > 0.25. Użyto maksimum globalnego: {best_height:.3f}")
    else:
        best_peak_idx = peaks[np.argmax(properties['peak_heights'])]
        best_height = corr_valid[best_peak_idx]
        dlog(f"Znaleziono {len(peaks)} pików. Główny pik: {best_height:.3f}")

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

    # Zapis pliku nagrania jeśli podano opcję save_rx lub debug
    target_rx = save_rx
    if debug and not target_rx:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target_rx = f"debug_rx_{timestamp}.wav"

    if target_rx:
        shutil.copyfile(TEMP_RX_WAV, target_rx)
        print(f"[ZAPIS AUDIO] Zapisano nagranie mikrofonu: {target_rx}")
        if debug:
            tx_save = target_rx.replace("debug_rx_", "debug_tx_") if "debug_rx_" in target_rx else "debug_tx.wav"
            shutil.copyfile(TEMP_TX_WAV, tx_save)
            print(f"[ZAPIS AUDIO] Zapisano wyemitowany sygnał: {tx_save}")

    dlog(f"Koniec pomiaru. Całkowity czas skryptu: {(time.perf_counter()-t_start)*1000.0:.1f} ms")

    # Zapis logów czasowych do pliku
    if debug:
        log_file = target_rx.rsplit(".", 1)[0] + ".log" if target_rx else f"debug_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"=== RAPORT CZASOWY SONAR TOF ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
            lf.write(f"Czas ToF: {tof_sec * 1000.0:.2f} ms | Dystans: {distance_calibrated_m * 100.0:.2f} cm\n\n")
            lf.write("\n".join(debug_logs) + "\n")
        print(f"[ZAPIS LOGÓW] Zapisano raport czasów etapów: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Akustyczny Sonar ToF dla Androida (Termux)")
    parser.add_argument("--fmin", type=float, default=2000, help="Początek ćwierku Hz (domyślnie 2000)")
    parser.add_argument("--fmax", type=float, default=8000, help="Koniec ćwierku Hz (domyślnie 8000)")
    parser.add_argument("--duration", type=float, default=0.080, help="Czas trwania w sekundach (domyślnie 0.08s)")
    parser.add_argument("--temp", type=float, default=21.0, help="Temperatura w °C (domyślnie 21.0)")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset kalibracyjny w metrach")
    parser.add_argument("--warmup", type=float, default=0.5, help="Czas trwania tonu wybudzającego Bluetooth w s (domyślnie 0.5s)")
    parser.add_argument("--echo", action="store_true", help="Tryb 2-way echa (odbicia od ściany)")
    parser.add_argument("--debug", action="store_true", help="Włącz szczegółowe logi czasowe kroków i automatyczny zapis plików audio")
    parser.add_argument("--save-rx", type=str, default=None, help="Ścieżka do zapisu nagranego pliku WAV z mikrofonu")
    args = parser.parse_args()

    if not check_termux_api():
        sys.exit(1)
    run_termux_sonar(
        f_start=args.fmin,
        f_end=args.fmax,
        chirp_duration=args.duration,
        temperature_c=args.temp,
        calib_offset_m=args.offset,
        two_way=args.echo,
        warmup_duration=args.warmup,
        debug=args.debug,
        save_rx=args.save_rx
    )


if __name__ == "__main__":
    main()
