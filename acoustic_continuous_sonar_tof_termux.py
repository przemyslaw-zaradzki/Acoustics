#!/usr/bin/env python3
"""
Akustyczny Sonar Ciągły (Continuous Tracking ToF) dla Termux (Android).
Metoda pomiaru z kalibracją odległości zerowej:
 1. W jednym, ciągłym nagraniu generowany jest ciąg jednakowych impulsów (chirpów)
    emitowanych w ściśle określonym interwale (np. co 300 ms).
 2. Pierwsze N impulsów (np. 5) służy jako faza kalibracji zerowej (głośnik leży przy smartfonie d = 0 cm).
 3. Po fazie kalibracji skrypt informuje użytkownika sygnałem tekstowym, że może przenieść głośnik.
 4. Kolejne impulsy mierzą przyrost opóźnienia względem fazy bazowej z kalibracji:
    Delta_d = (Delta_t_odbioru) * c.
 5. Brak przerywania odtwarzania i nagrywania eliminuje jitter buforowania Bluetooth A2DP.

python acoustic_continuous_sonar_tof_termux.py --warmup 0.6 --interval 0.35 --n-calib 6 --n-meas 25
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

TMP_DIR = os.environ.get("TMPDIR", os.environ.get("PREFIX", "") + "/tmp" if os.environ.get("PREFIX") else tempfile.gettempdir())
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR, exist_ok=True)

TEMP_TX_WAV = os.path.join(TMP_DIR, "continuous_sonar_tx.wav")
TEMP_RX_RAW = os.path.join(TMP_DIR, "continuous_sonar_rx.m4a")
TEMP_RX_WAV = os.path.join(TMP_DIR, "continuous_sonar_rx.wav")


def check_termux_api():
    """Sprawdza dostępność niezbędnych poleceń w Termux."""
    cmds = ["termux-microphone-record", "termux-media-player", "ffmpeg"]
    missing = [c for c in cmds if shutil.which(c) is None]
    if missing:
        print("\n" + "!" * 60)
        print("[BŁĄD] Brakujące polecenia systemowe:", ", ".join(missing))
        print("Aby je zainstalować, wykonaj w konsoli Termux:")
        print("    pkg install termux-api ffmpeg")
        print("!" * 60 + "\n")
        return False
    return True


def calculate_speed_of_sound(temperature_c=21.0):
    """Oblicza prędkość dźwięku w m/s."""
    return 331.3 * np.sqrt(1.0 + temperature_c / 273.15)


def generate_continuous_tx(
    filename,
    f_start=2000,
    f_end=8000,
    chirp_duration=0.040,
    interval=0.300,
    n_calib=5,
    n_meas=25,
    fs=44100,
    warmup_duration=0.6,
    warmup_freq=440.0,
    warmup_gain=0.03
):
    """
    Tworzy ciągły plik audio zawierający:
    1. Ton wybudzający (warmup) dla głośnika Bluetooth.
    2. Przerwę wyciszenia (0.2 s).
    3. Serię chirpów emitowanych z równym odstępem czasu (interval):
       - n_calib impulsów kalibracyjnych (pozycja 0 cm)
       - n_meas impulsów właściwego pomiaru (po przeniesieniu głośnika)
    """
    total_chirps = n_calib + n_meas
    interval_samples = int(fs * interval)
    chirp_samples = int(fs * chirp_duration)

    # 1. Warm-up
    if warmup_duration > 0:
        n_warmup = int(fs * warmup_duration)
        t_w = np.linspace(0, warmup_duration, n_warmup, endpoint=False)
        w_window = signal.windows.tukey(n_warmup, alpha=0.1)
        warmup_sig = (np.sin(2 * np.pi * warmup_freq * t_w) * warmup_gain * w_window).astype(np.float32)
    else:
        warmup_sig = np.array([], dtype=np.float32)

    # 2. Wzorzec chirpu
    t_c = np.linspace(0, chirp_duration, chirp_samples, endpoint=False)
    chirp_raw = signal.chirp(t_c, f0=f_start, t1=chirp_duration, f1=f_end, method='linear')
    window = signal.windows.tukey(chirp_samples, alpha=0.1)
    chirp = (chirp_raw * window).astype(np.float32)

    # 3. Złożenie całego strumienia
    silence_gap = int(fs * 0.2)
    tx_parts = []
    if len(warmup_sig) > 0:
        tx_parts.append(warmup_sig)
    tx_parts.append(np.zeros(silence_gap, dtype=np.float32))

    start_chirps_sample = sum(len(p) for p in tx_parts)

    # Pętla generowania bloków o długości interval_samples
    for i in range(total_chirps):
        block = np.zeros(interval_samples, dtype=np.float32)
        block[:chirp_samples] = chirp
        tx_parts.append(block)

    # Dodatkowa sekunda ciszy na końcu, by nie uciąć wybrzmienia
    tx_parts.append(np.zeros(fs, dtype=np.float32))

    tx_signal = np.concatenate(tx_parts)
    audio_int16 = np.int16(np.clip(tx_signal, -1.0, 1.0) * 32767)
    wavfile.write(filename, fs, audio_int16)

    total_time_s = len(tx_signal) / fs
    calib_time_s = (start_chirps_sample + n_calib * interval_samples) / fs

    return chirp, fs, interval_samples, total_chirps, total_time_s, calib_time_s


def run_continuous_sonar(
    f_start=2000,
    f_end=8000,
    chirp_duration=0.040,
    interval=0.300,
    n_calib=5,
    n_meas=25,
    temperature_c=21.0,
    warmup_duration=0.6,
    debug=False,
    save_rx=None
):
    c = calculate_speed_of_sound(temperature_c)
    fs = 44100
    t_start = time.perf_counter()

    print("\n" + "="*65)
    print("      SONAR CIĄGŁY Z KALIBRACJĄ ZEROWĄ (BLUETOOTH / TERMUX)")
    print("="*65)
    print(f"[*] Prędkość dźwięku (T={temperature_c:.1f}°C): c = {c:.2f} m/s")
    print(f"[*] Parametry: pasmo={f_start}-{f_end} Hz, interwał={interval*1000:.0f} ms")
    print(f"[*] Impulsy: {n_calib} do kalibracji (0 cm), {n_meas} pomiarowych")

    # 1. Generacja pliku audio
    chirp_pattern, fs, interval_samples, total_chirps, total_time_s, calib_time_s = generate_continuous_tx(
        TEMP_TX_WAV,
        f_start=f_start,
        f_end=f_end,
        chirp_duration=chirp_duration,
        interval=interval,
        n_calib=n_calib,
        n_meas=n_meas,
        fs=fs,
        warmup_duration=warmup_duration
    )

    # 2. Czyszczenie starych plików
    for f in [TEMP_RX_RAW, TEMP_RX_WAV]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    print("\n" + "#"*65)
    print(">> INSTRUKCJA: UMIEŚĆ GŁOŚNIK GO3 BEZPOŚREDNIO PRZY TELEFONIE (0 cm) <<")
    print("#"*65 + "\n")
    print("[*] Rozpoczynanie sesji pomiarowej...")

    # 3. Uruchomienie mikrofonu
    rec_proc = subprocess.Popen(
        ["termux-microphone-record", "-f", TEMP_RX_RAW, "-e", "aac", "-r", "44100"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(0.3)

    # 4. Uruchomienie odtwarzacza
    t_play_cmd = time.perf_counter()
    subprocess.run(["termux-media-player", "play", TEMP_TX_WAV], capture_output=True, check=False)

    # Czas do zakończenia fazy kalibracji
    # Czekamy na warmup + czas trwania kalibracji
    time.sleep(calib_time_s + 0.5)

    print("\n" + "!"*65)
    print(">> KALIBRACJA 0 cm ZAKOŃCZONA! MOŻESZ TERAZ PRZENIEŚĆ GŁOŚNIK GO3 <<")
    print(f">> Masz około {(n_meas * interval):.1f} sekund na pomiary w zadanym miejscu <<")
    print("!"*65 + "\n")

    # Czekamy na resztę emisji
    remaining_time = max(0.0, total_time_s - calib_time_s + 0.8)
    time.sleep(remaining_time)

    # 5. Zatrzymanie nagrywania
    print("[*] Zatrzymywanie nagrywania mikrofonem...")
    subprocess.run(["termux-microphone-record", "-q"], check=False)
    try:
        rec_proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        rec_proc.kill()

    time.sleep(0.4)

    # 6. Sprawdzenie i konwersja ffmpeg
    for _ in range(25):
        if os.path.exists(TEMP_RX_RAW) and os.path.getsize(TEMP_RX_RAW) > 1000:
            break
        time.sleep(0.1)

    if not os.path.exists(TEMP_RX_RAW) or os.path.getsize(TEMP_RX_RAW) == 0:
        print("[BŁĄD] Plik nagrania nie powstał lub jest pusty. Sprawdź uprawnienia mikrofonu Termux:API.")
        return

    conv = subprocess.run(
        ["ffmpeg", "-y", "-i", TEMP_RX_RAW, "-ar", str(fs), "-ac", "1", TEMP_RX_WAV],
        capture_output=True,
        text=True
    )
    if conv.returncode != 0 or not os.path.exists(TEMP_RX_WAV):
        print("[BŁĄD] Błąd konwersji nagrania przez ffmpeg.")
        return

    try:
        sample_rate, rx_raw = wavfile.read(TEMP_RX_WAV)
    except Exception as e:
        print(f"[BŁĄD] Nie można odczytać nagrania WAV: {e}")
        return

    if rx_raw.ndim > 1:
        rx_raw = rx_raw.mean(axis=1)
    rx_signal = rx_raw.astype(np.float32) / 32768.0

    print(f"[*] Wczytano nagranie: {len(rx_signal)/sample_rate:.2f} s ({len(rx_signal)} próbek)")

    # 7. Filtracja pasmowa
    nyquist = 0.5 * sample_rate
    low_cut = max((f_start * 0.8) / nyquist, 0.01)
    high_cut = min((f_end * 1.2) / nyquist, 0.99)
    b, a = signal.butter(4, [low_cut, high_cut], btype='band')
    rx_filtered = signal.filtfilt(b, a, rx_signal)

    # 8. Korelacja wzajemna (filtr dopasowany)
    correlation = signal.correlate(rx_filtered, chirp_pattern, mode='valid', method='fft')
    corr_envelope = np.abs(signal.hilbert(correlation))
    max_c = np.max(corr_envelope)
    if max_c > 0:
        corr_envelope /= max_c

    # 9. Wyszukanie sekwencji chirpów
    # Szukamy pików z minimalnym odstępem około 80% zadanego interwału
    min_dist_samples = int(interval_samples * 0.75)
    peaks, props = signal.find_peaks(corr_envelope, height=0.20, distance=min_dist_samples)

    if len(peaks) < n_calib:
        print(f"[OSTRZEŻENIE] Wykryto tylko {len(peaks)} pików (oczekiwano min. {n_calib} do kalibracji).")
        print("Sygnał mógł być zbyt cichy lub zaszumiony.")
        if len(peaks) == 0:
            return

    print(f"[*] Łącznie wykryto {len(peaks)} impulsów akustycznych w nagraniu.")

    # 10. Kalibracja zerowej odległości
    # Wyznaczamy bazową fazę piku z pierwszych n_calib impulsów
    calib_peaks = peaks[:n_calib]
    # Przewidywany interwał idealny to interval_samples
    # Z piku kalibracyjnego k: przewidywana pozycja piku 0 to: calib_peaks[k] - k * interval_samples
    t0_estimates = [calib_peaks[k] - k * interval_samples for k in range(len(calib_peaks))]
    t0_base_sample = np.mean(t0_estimates)

    print("\n" + "="*65)
    print("                   WYNIKI KALIBRACJI ZEROWEJ (0 cm)")
    print("="*65)
    for idx, p in enumerate(calib_peaks):
        expected_p = t0_base_sample + idx * interval_samples
        diff_samples = p - expected_p
        diff_ms = (diff_samples / fs) * 1000.0
        print(f"  Impuls kalibracyjny #{idx+1}: próbka {p} (odchyłka: {diff_ms:+6.2f} ms | {corr_envelope[p]:.2f})")

    # 11. Obliczenie odległości dla kolejnych impulsów (faza właściwa)
    print("\n" + "="*65)
    print("                  POMIARY WŁAŚCIWE (PO KALIBRACJI)")
    print("="*65)
    print(" Nr | Czas nagrania | Względny ToF | Zmierzona Odległość | Jakość piku")
    print("-" * 65)

    distances = []
    for idx in range(n_calib, len(peaks)):
        p = peaks[idx]
        # Przewidywana pozycja, gdyby głośnik nadal stał w punkcie 0 cm (przy nominalnym zegarze)
        nominal_p0 = t0_base_sample + idx * interval_samples
        # Dodatkowe opóźnienie wywołane odsunięciem głośnika
        delta_samples = p - nominal_p0
        delta_t_sec = delta_samples / fs
        dist_m = delta_t_sec * c

        rec_time_s = p / fs
        peak_quality = corr_envelope[p]

        # Filtrujemy tylko odczyty z fazy po odstawieniu (np. ostatnie impulsy)
        distances.append(dist_m)
        print(f" #{idx+1:02d} | {rec_time_s:6.2f} s      | {delta_t_sec*1000:+8.2f} ms | {dist_m*100:8.2f} cm ({dist_m:5.2f} m) | {peak_quality:.2f}")

    if distances:
        # Weźmy stabilne odczyty z końcówki (np. ostatnie 5 impulsów gdy głośnik już stał w miejscu)
        last_samples = distances[-min(5, len(distances)):]
        avg_dist = np.mean(last_samples)
        std_dist = np.std(last_samples)
        print("\n" + "="*65)
        print(f"  PODSUMOWANIE (Średnia z ostatnich {len(last_samples)} pomiarów, gdy głośnik stał):")
        print(f"  Średnia odległość: {avg_dist*100.0:.1f} cm ({avg_dist:.2f} m)")
        print(f"  Rozrzut (std dev):  {std_dist*100.0:.1f} cm")
        print("="*65 + "\n")

    # Zapis plików debug jeśli zażądano
    if debug:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_rx = save_rx if save_rx else f"debug_continuous_rx_{timestamp}.wav"
        out_tx = f"debug_continuous_tx_{timestamp}.wav"
        shutil.copyfile(TEMP_RX_WAV, out_rx)
        shutil.copyfile(TEMP_TX_WAV, out_tx)
        print(f"[ZAPIS AUDIO] Zapisano nagranie: {out_rx}")
        print(f"[ZAPIS AUDIO] Zapisano nadany sygnał: {out_tx}")


def main():
    parser = argparse.ArgumentParser(description="Akustyczny Sonar Ciągły z kalibracją odległości zerowej dla Termux")
    parser.add_argument("--fmin", type=float, default=2000, help="Początek ćwierku Hz (domyślnie 2000)")
    parser.add_argument("--fmax", type=float, default=8000, help="Koniec ćwierku Hz (domyślnie 8000)")
    parser.add_argument("--duration", type=float, default=0.040, help="Czas trwania pojedynczego chirpu w s (domyślnie 0.04s)")
    parser.add_argument("--interval", type=float, default=0.350, help="Odstęp między chirpami w s (domyślnie 0.35s)")
    parser.add_argument("--n-calib", type=int, default=6, help="Liczba impulsów kalibracyjnych przy telefonie (domyślnie 6)")
    parser.add_argument("--n-meas", type=int, default=25, help="Liczba impulsów właściwych (domyślnie 25)")
    parser.add_argument("--temp", type=float, default=21.0, help="Temperatura w °C (domyślnie 21.0)")
    parser.add_argument("--warmup", type=float, default=0.6, help="Czas trwania tonu wybudzającego Bluetooth w s (domyślnie 0.6s)")
    parser.add_argument("--debug", action="store_true", help="Zapisz pliki audio z sesji")
    parser.add_argument("--save-rx", type=str, default=None, help="Ścieżka do zapisu pliku WAV z nagrania")
    args = parser.parse_args()

    if not check_termux_api():
        sys.exit(1)

    run_continuous_sonar(
        f_start=args.fmin,
        f_end=args.fmax,
        chirp_duration=args.duration,
        interval=args.interval,
        n_calib=args.n_calib,
        n_meas=args.n_meas,
        temperature_c=args.temp,
        warmup_duration=args.warmup,
        debug=args.debug,
        save_rx=args.save_rx
    )


if __name__ == "__main__":
    main()

