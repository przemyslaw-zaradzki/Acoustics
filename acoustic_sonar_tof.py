#!/usr/bin/env python3
"""
Eksperyment 1: Akustyczny Sonar ToF (Time-of-Flight)
Pomiar odległości za pomocą ultradźwięków (ćwierk / chirp 18 kHz - 21 kHz)

Działa z laptopem oraz głośnikiem zewnętrznym (np. JBL GO 3 przez Bluetooth / AUX / USB)
albo wbudowanymi głośnikami laptopa.
"""

import sys
import os

# Ustawienie katalogu cache matplotlib jeśli domyślny jest tylko do odczytu
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import argparse
import time
import numpy as np
from scipy import signal
import matplotlib
try:
    import matplotlib.pyplot as plt
except Exception:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

try:
    import sounddevice as sd
except ImportError:
    print("\n[BŁĄD] Biblioteka 'sounddevice' nie jest zainstalowana.")
    print("Zainstaluj wymagane pakiety wykonując:")
    print("    pip install sounddevice numpy scipy matplotlib\n")
    sys.exit(1)


def list_audio_devices():
    """Wypisuje dostępne urządzenia wejściowe i wyjściowe audio."""
    print("\n=== Dostępne Urządzenia Audio ===")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        in_ch = dev['max_input_channels']
        out_ch = dev['max_output_channels']
        default_in = " [Domyślne WE]" if idx == sd.default.device[0] else ""
        default_out = " [Domyślne WY]" if idx == sd.default.device[1] else ""
        print(f"[{idx:2d}] {dev['name']} (Wejścia: {in_ch}, Wyjścia: {out_ch}){default_in}{default_out}")
    print("=================================\n")


def generate_ultrasonic_chirp(f_start=2000, f_end=8000, duration=0.080, fs=48000):
    """
    Generuje ćwierk (chirp) z nałożonym oknem Hanninga/Tukeya.
    
    Domyślnie: 2 kHz - 8 kHz, czas trwania 80 ms.
    Pasmo słyszalne zapewnia potężną energię akustyczną, brak tłumienia przez
    filtry antysprzężeniowe mikrofonu laptopa i odporność na jitter.
    
    :param f_start: Początkowa częstotliwość (Hz)
    :param f_end: Końcowa częstotliwość (Hz)
    :param duration: Czas trwania ćwierku (s)
    :param fs: Częstotliwość próbkowania (Hz)
    :return: (t, chirp_signal)
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    chirp_raw = signal.chirp(t, f0=f_start, t1=duration, f1=f_end, method='linear')
    # Okno Tukey 10% na brzegach zapobiegające trzaskom
    window = signal.windows.tukey(len(chirp_raw), alpha=0.1)
    chirp_windowed = chirp_raw * window
    return t, chirp_windowed.astype(np.float32)


def calculate_speed_of_sound(temperature_c=20.0):
    """Oblicza prędkość dźwięku w powietrzu dla danej temperatury w °C."""
    return 331.3 * np.sqrt(1.0 + temperature_c / 273.15)


def run_sonar_measurement(
    output_device=None,
    input_device=None,
    fs=48000,
    chirp_duration=0.080,
    rec_duration=0.35,
    f_start=2000,
    f_end=8000,
    temperature_c=20.0,
    two_way=False,
    calib_offset_m=0.0,
    plot_result=True,
    output_image="sonar_tof_result.png"
):
    """
    Wykonuje pomiar odległości metodą korelacji wzajemnej sygnału ćwierku.
    
    :param two_way: True jeśli fala leci do przeszkody i wraca do mikrofonu (echo),
                    False jeśli mierzymy odległość między głośnikiem a mikrofonem (przelot bezpośredni 1-way).
    :param calib_offset_m: Kalibracja/offset sprzętowy w metrach (np. opóźnienie bufora karty dźwiękowej).
    """
    c = calculate_speed_of_sound(temperature_c)
    print(f"[*] Prędkość dźwięku przy {temperature_c:.1f}°C: c = {c:.2f} m/s")
    print(f"[*] Rozdzielczość jednej próbki (Δd = c / fs): {1000.0 * c / fs:.2f} mm")
    print(f"[*] Zakres ćwierku: {f_start} Hz - {f_end} Hz (czas trwania: {chirp_duration*1000:.1f} ms)")

    # 1. Przygotowanie sygnału nadawczego
    _, chirp = generate_ultrasonic_chirp(f_start, f_end, chirp_duration, fs)

    # Przygotowujemy bufor odtwarzania i nagrywania
    # Dodajemy krótką ciszę przed ćwierkiem (np. 10 ms), aby karta dźwiękowa zdążyła się ustabilizować
    pre_silence_samples = int(fs * 0.010)
    post_silence_samples = int(fs * (rec_duration - chirp_duration - 0.010))
    if post_silence_samples < 0:
        post_silence_samples = int(fs * 0.05)

    tx_signal = np.pad(chirp, (pre_silence_samples, post_silence_samples), mode='constant')
    rec_samples = len(tx_signal)

    print(f"[*] Nadawanie i jednoczesne nagrywanie ({rec_samples / fs:.2f} s)...")
    
    # 2. Równoczesne odtwarzanie i nagrywanie
    try:
        rx_signal = sd.playrec(
            tx_signal,
            samplerate=fs,
            channels=1,
            device=(input_device, output_device),
            dtype='float32'
        )
        sd.wait()
    except Exception as e:
        print(f"\n[BŁĄD] Wystąpił błąd podczas komunikacji z kartą dźwiękową:")
        print(e)
        return

    rx_signal = rx_signal.flatten()

    # 3. Filtracja pasmowoprzepustowa nagranego sygnału
    nyquist = 0.5 * fs
    low_cut = max((f_start * 0.8) / nyquist, 0.01)
    high_cut = min((f_end * 1.2) / nyquist, 0.99)
    b, a = signal.butter(4, [low_cut, high_cut], btype='band')
    rx_filtered = signal.filtfilt(b, a, rx_signal)

    # 4. Korelacja wzajemna (filtr dopasowany / matched filter)
    # Wykorzystujemy FFT dla maksymalnej wydajności
    correlation = signal.correlate(rx_filtered, chirp, mode='valid', method='fft')
    corr_envelope = np.abs(signal.hilbert(correlation))  # Obwiednia analityczna (Hilbert)
    corr_envelope /= (np.max(corr_envelope) + 1e-12)     # Normalizacja 0 - 1

    # Wektor czasu dla korelacji
    # Pierwsza próbka korelacji odpowiada dopasowaniu na początku nagrania
    time_axis = np.arange(len(corr_envelope)) / fs
    # Odejmujemy margines początkowej ciszy nadawczej
    time_flight = time_axis - (pre_silence_samples / fs)

    # 5. Detekcja pików
    # Szukamy piku o największej korelacji (lub pierwszego znaczącego piku)
    # Odcinamy ujemne czasy wynikające z ciszy
    valid_mask = time_flight >= 0
    t_valid = time_flight[valid_mask]
    corr_valid = corr_envelope[valid_mask]

    peaks, properties = signal.find_peaks(corr_valid, height=0.25, distance=int(fs * 0.001))
    
    if len(peaks) == 0:
        print("[!] Nie wykryto wyraźnego echa ćwierku. Spróbuj zwiększyć głośność lub przybliżyć mikrofon/głośnik.")
        best_peak_idx = np.argmax(corr_valid)
    else:
        # Najsilniejszy pik
        best_peak_idx = peaks[np.argmax(properties['peak_heights'])]

    tof_sec = t_valid[best_peak_idx]
    
    # Obliczenie odległości
    divisor = 2.0 if two_way else 1.0
    distance_raw_m = (tof_sec * c) / divisor
    distance_calibrated_m = max(0.0, distance_raw_m - calib_offset_m)

    mode_str = "Dojście bezpośrednie (1-way: głośnik -> mikrofon)" if not two_way else "Echo / Odbicie (2-way: głośnik -> przeszkoda -> mikrofon)"
    print("\n" + "="*50)
    print(f"  WYNIK POMIARU SONARU ToF ({mode_str})")
    print("="*50)
    print(f"  Czas przelotu (ToF):      {tof_sec * 1000.0:.3f} ms ({tof_sec * 1e6:.1f} µs)")
    print(f"  Zmierzona odległość:      {distance_calibrated_m * 100.0:.2f} cm ({distance_calibrated_m:.3f} m)")
    if calib_offset_m > 0:
        print(f"  (Uwzględniono offset:     -{calib_offset_m * 100.0:.2f} cm)")
    print("="*50 + "\n")

    # Wypisz kolejne echa (odbicia od ścian/obiektów)
    if len(peaks) > 1:
        print("Wykryte echa / odbicia wielodrogowe:")
        for i, p in enumerate(peaks):
            p_time = t_valid[p]
            p_dist = (p_time * c) / divisor - calib_offset_m
            p_amp = corr_valid[p]
            marker = " <-- GŁÓWNY SYGNAŁ" if p == best_peak_idx else ""
            print(f"  Pik #{i+1}: t = {p_time*1000.0:6.2f} ms | d = {p_dist*100.0:6.2f} cm | korelacja = {p_amp:.2f}{marker}")
        print()

    # 6. Wykres i wizualizacja
    if plot_result:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

        # Wykres 1: Sygnał nadawany
        axs[0].plot(tx_signal[:int(fs * 0.04)] * 1000, color='tab:blue')
        axs[0].set_title("Sygnał Nadawany (Chirp ultradźwiękowy 18-21 kHz, zoom na pierwsze 40 ms)")
        axs[0].set_xlabel("Próbki")
        axs[0].set_ylabel("Amplituda [mFS]")
        axs[0].grid(True, linestyle='--', alpha=0.6)

        # Wykres 2: Sygnał odebrany (przefiltrowany mikrofon)
        axs[1].plot(np.arange(len(rx_filtered)) / fs * 1000.0, rx_filtered, color='tab:orange')
        axs[1].set_title("Sygnał Odebrany z Mikrofonu (po filtracji 17-22 kHz)")
        axs[1].set_xlabel("Czas [ms]")
        axs[1].set_ylabel("Amplituda")
        axs[1].grid(True, linestyle='--', alpha=0.6)

        # Wykres 3: Korelacja wzajemna (profil opóźnień i echa)
        dist_axis_cm = ((t_valid * c) / divisor - calib_offset_m) * 100.0
        # Ograniczamy wykres do sensownego zasięgu, np. do 3 metrów
        plot_limit = dist_axis_cm <= 300.0
        axs[2].plot(dist_axis_cm[plot_limit], corr_valid[plot_limit], color='tab:green', label='Obwiednia korelacji')
        axs[2].scatter([distance_calibrated_m * 100.0], [corr_valid[best_peak_idx]], color='red', s=50, zorder=5,
                        label=f'Wykryty pik: {distance_calibrated_m*100.0:.1f} cm')
        axs[2].axvline(distance_calibrated_m * 100.0, color='red', linestyle='--', alpha=0.7)
        axs[2].set_title(f"Filtr Dopasowany (Korelacja Wzajemna) -> Odległość: {distance_calibrated_m*100.0:.1f} cm")
        axs[2].set_xlabel("Odległość [cm]")
        axs[2].set_ylabel("Współczynnik korelacji")
        axs[2].set_ylim(-0.05, 1.05)
        axs[2].legend(loc='upper right')
        axs[2].grid(True, linestyle='--', alpha=0.6)

        plt.savefig(output_image, dpi=200)
        print(f"[+] Zapisano wykres do pliku: {output_image}")
        try:
            # Jeśli jest środowisko graficzne, spróbuj wyświetlić
            if "DISPLAY" in os.environ:
                plt.show()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Akustyczny Sonar ToF (Time-of-Flight) - Pomiar odległości ultradźwiękami (JBL + Mikrofon laptopa)"
    )
    parser.add_argument("--list-devices", action="store_true", help="Wyświetl listę dostępnych urządzeń audio")
    parser.add_argument("--out-dev", type=int, default=None, help="Numer urządzenia wyjściowego (np. głośnik JBL)")
    parser.add_argument("--in-dev", type=int, default=None, help="Numer urządzenia wejściowego (np. mikrofon laptopa)")
    parser.add_argument("--temp", type=float, default=21.0, help="Temperatura otoczenia w °C (domyślnie 21.0)")
    parser.add_argument("--fmin", type=float, default=2000, help="Początkowa częstotliwość ćwierku w Hz (domyślnie 2000)")
    parser.add_argument("--fmax", type=float, default=8000, help="Końcowa częstotliwość ćwierku w Hz (domyślnie 8000)")
    parser.add_argument("--duration", type=float, default=0.080, help="Czas trwania ćwierku w sekundach (domyślnie 0.080 s = 80 ms)")
    parser.add_argument("--echo", action="store_true", help="Włącz tryb echa 2-way (odbicie od ściany). Domyślnie tryb 1-way bezpośredni.")
    parser.add_argument("--one-way", action="store_true", help="Tryb 1-way bezpośredni (domyślny)")
    parser.add_argument("--offset", type=float, default=0.0, help="Kalibracja przesunięcia w metrach (np. opóźnienie bufora)")
    parser.add_argument("--save-plot", type=str, default="sonar_tof_result.png", help="Ścieżka do zapisu wykresu PNG")
    parser.add_argument("--loop", action="store_true", help="Tryb ciągłego radaru / sonaru (pomiar co sekundę w pętli)")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    print("==========================================================")
    print("  Akustyczny Sonar ToF - Pomiar Dźwiękowy (DIY)           ")
    print("==========================================================")
    print("Wskazówka: Uruchom z opcją --list-devices, aby wybrać głośnik JBL oraz mikrofon.")
    
    # Domyślnie 1-way (bezpośredni lot głośnik -> laptop)
    two_way = args.echo

    if args.loop:
        print("[*] Uruchamianie w trybie ciągłym (Ctrl+C, aby przerwać)...")
        try:
            while True:
                run_sonar_measurement(
                    output_device=args.out_dev,
                    input_device=args.in_dev,
                    chirp_duration=args.duration,
                    f_start=args.fmin,
                    f_end=args.fmax,
                    temperature_c=args.temp,
                    two_way=two_way,
                    calib_offset_m=args.offset,
                    plot_result=False
                )
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[*] Zakończono pomiary ciągłe.")
    else:
        run_sonar_measurement(
            output_device=args.out_dev,
            input_device=args.in_dev,
            chirp_duration=args.duration,
            f_start=args.fmin,
            f_end=args.fmax,
            temperature_c=args.temp,
            two_way=two_way,
            calib_offset_m=args.offset,
            plot_result=True,
            output_image=args.save_plot
        )


if __name__ == "__main__":
    main()
