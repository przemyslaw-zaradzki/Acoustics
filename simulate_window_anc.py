"""
Symulacja aktywnego tłumienia hałasu w oknie (Anti-Noise Window)
z wykorzystaniem zdecentralizowanych jednostek feedback ANC w bibliotece pyroomacoustics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra


def simulate_anti_noise_window(n_units=5):
    fs = 8000  # Częstotliwość próbkowania [Hz]
    c = 343.0  # Prędkość dźwięku [m/s]
    freq_noise = 150.0  # Częstotliwość hałasu (np. dudnienie silnika / ruch uliczny) [Hz]
    duration = 1.0  # Czas trwania analizowanego sygnału [s]
    
    # ---------------------------------------------------------
    # 1. Konfiguracja geometrii: Pokój z oknem
    # Pokój o wymiarach 5m x 4m (x: 0..5, y: 0..4)
    # Okno znajduje się na dolnej ścianie y=0, w zakresie x od 1.5m do 3.5m (szerokość 2.0 m)
    # ---------------------------------------------------------
    room_dim = [5.0, 4.0]
    
    # Współczynniki pochłaniania ścian pokoju (standardowy umeblowany pokój)
    absorption = 0.15
    max_order = 3  # Rząd odbić dla Image Source Method

    # Zewnętrzne źródło hałasu (np. samochód na ulicy, 2 metry od okna)
    noise_pos = [2.5, -2.0]  # x=2.5, y=-2.0 (poza pokojem)
    
    # Modelujemy okno jako aperturę:
    # Fala ze źródła na zewnątrz dociera do płaszczyzny okna (y=0) z opóźnieniem i tłumieniem sferycznym.
    # W świetle okna rozmieszczamy N_units jednostek ANC (głośnik + mikrofon).
    window_x = np.linspace(1.7, 3.3, n_units)
    window_y = 0.05  # Tuż w otworze okna od strony wnętrza
    
    speaker_positions = np.zeros((2, n_units))
    speaker_positions[0, :] = window_x
    speaker_positions[1, :] = window_y

    # Mikrofony kolokowane z głośnikami (np. 3 cm od głośnika w stronę zewnętrzną)
    mic_feedback_positions = np.zeros((2, n_units))
    mic_feedback_positions[0, :] = window_x
    mic_feedback_positions[1, :] = window_y - 0.03  # ku zewnątrz

    # Siatka mikrofonów pomiarowych wewnątrz pokoju do oceny pola akustycznego i energii
    grid_x = np.linspace(0.5, 4.5, 25)
    grid_y = np.linspace(0.5, 3.5, 20)
    GX, GY = np.meshgrid(grid_x, grid_y)
    eval_mics = np.vstack([GX.ravel(), GY.ravel()])  # Kształt: (2, N_mics)

    # ---------------------------------------------------------
    # 2. Obliczenia w dziedzinie częstotliwości (Helmholtz / Green's functions)
    # Dla fali harmonicznej p(t) = Re{ P * exp(j * omega * t) }
    # ---------------------------------------------------------
    omega = 2 * np.pi * freq_noise
    k = omega / c  # Liczba falowa

    # A. Ciśnienie z zewnętrznego źródła na mikrofony feedback w oknie
    # Swobodna propagacja 2D/3D z zewnątrz do okna:
    dist_noise_to_fb = np.linalg.norm(mic_feedback_positions.T - noise_pos, axis=1)
    # Fala sferyczna ze źródła: P_inc = A / r * exp(-j * k * r)
    P_inc_window = (1.0 / dist_noise_to_fb) * np.exp(-1j * k * dist_noise_to_fb)

    # B. Wyznaczenie transmitancji wewnątrz pokoju za pomocą pyroomacoustics
    # Tworzymy pokój w pyroomacoustics do symulacji rozchodzenia się dźwięku wewnątrz
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption), max_order=max_order)

    # Dodajemy źródła wtórne (głośniki w oknie)
    for i in range(n_units):
        room.add_source(speaker_positions[:, i])

    # Dodajemy także zastępcze źródło reprezentujące wpadający hałas z zewnątrz
    # (dla uproszczenia jako falę wnikającą przez środek okna)
    room.add_source([2.5, 0.05])
    noise_virtual_idx = n_units  # Indeks tego źródła

    # Dodajemy siatkę mikrofonów do pokoju
    room.add_microphone_array(pra.MicrophoneArray(eval_mics, fs=fs))
    room.compute_rir()

    # Wyliczamy transmitancje H(f) z RIR dla częstotliwości freq_noise
    n_eval_mics = eval_mics.shape[1]
    H_speakers = np.zeros((n_eval_mics, n_units), dtype=complex)
    H_noise = np.zeros(n_eval_mics, dtype=complex)

    fft_len = 4096
    freq_bins = np.fft.rfftfreq(fft_len, 1.0 / fs)
    target_bin = np.argmin(np.abs(freq_bins - freq_noise))

    for m in range(n_eval_mics):
        # Transmitancja od głośników ANC do punktów w pokoju
        for s in range(n_units):
            rir = room.rir[m][s]
            H_fft = np.fft.rfft(rir, n=fft_len)
            H_speakers[m, s] = H_fft[target_bin]
        
        # Transmitancja od wpadającego hałasu do punktów w pokoju
        rir_n = room.rir[m][noise_virtual_idx]
        H_noise[m] = np.fft.rfft(rir_n, n=fft_len)[target_bin]

    # ---------------------------------------------------------
    # 3. Model zdecentralizowanego sterowania sprzężeniem zwrotnym (Feedback ANC)
    # Każda jednostka dąży do zerowania ciśnienia w swoim punkcie (Direct Velocity/Pressure Feedback)
    # Warunek zerowania ciśnienia lokalnego: P_inc + S_local * Q_s = 0 => Q_s = - P_inc / S_local
    # Uwzględniamy stabilne wzmocnienie pętli Gain (np. 80-90% idealnego tłumienia)
    # ---------------------------------------------------------
    gain_factor = 3 * 0.88 / n_units  # Wzmocnienie pętli (stabilność układu feedback)
    phase_shift = np.pi  # Przeciwfaza (antyhałas)
    
    # Sygnały wyemitowane przez głośniki wtórne
    speaker_weights = gain_factor * np.exp(1j * phase_shift) * P_inc_window

    # ---------------------------------------------------------
    # 4. Obliczenie ciśnienia akustycznego i energii przed i po włączeniu ANC
    # ---------------------------------------------------------
    # Ciśnienie bez ANC (sam hałas)
    P_off = H_noise * 1.0  # znormalizowana amplituda hałasu

    # Ciśnienie z ANC (hałas + suma fal z głośników przeciwstawnych)
    P_anc = np.sum(H_speakers * speaker_weights[np.newaxis, :], axis=1)
    P_on = P_off + P_anc

    # Gęstość energii akustycznej ~ |P|^2
    energy_off = np.mean(np.abs(P_off) ** 2)
    energy_on = np.mean(np.abs(P_on) ** 2)

    # Globalna redukcja energii
    energy_reduction_pct = (1.0 - energy_on / energy_off) * 100
    global_reduction_db = 10 * np.log10(energy_on / energy_off)

   
    log_message = f"""============================================================
    WYNIKI SYMULACJI ANTI-NOISE WINDOW ({freq_noise:.0f} Hz):
    Liczba jednostek feedback w oknie: {n_units}
    Średnia energia akustyczna bez ANC: {energy_off:.5e}
    Średnia energia akustyczna z ANC:   {energy_on:.5e}
    Procent pochłoniętej/zredukowanej energii: {energy_reduction_pct:.2f}%
    Średnia redukcja poziomu ciśnienia w pokoju: {global_reduction_db:.2f} dB
    ============================================================"""

    # Wypisanie w terminalu
    print(log_message)

    # Zapis do pliku (tryb 'a' dopisuje na końcu pliku, 'w' nadpisuje plik)
    with open(f"anti_noise_window_result_{n_units}.log", "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


    # ---------------------------------------------------------
    # 5. Wizualizacja: Rozkład poziomu dźwięku SPL [dB] w pokoju
    # ---------------------------------------------------------
    spl_off = 10 * np.log10(np.abs(P_off) ** 2 + 1e-12)
    spl_on = 10 * np.log10(np.abs(P_on) ** 2 + 1e-12)
    diff_spl = spl_on - spl_off  # Spadek ciśnienia (wartości ujemne = wyciszenie)

    Z_diff = diff_spl.reshape(GX.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mapa bez ANC vs z ANC
    im1 = axes[0].contourf(GX, GY, spl_off.reshape(GX.shape), levels=30, cmap='inferno')
    axes[0].set_title("Rozkład ciśnienia akustycznego BEZ ANC")
    axes[0].set_xlabel("Szerokość pokoju [m]")
    axes[0].set_ylabel("Długość pokoju [m]")
    fig.colorbar(im1, ax=axes[0], label="Względny poziom SPL [dB]")
    # Zaznacz okno
    axes[0].plot([1.5, 3.5], [0, 0], 'c-', linewidth=6, label="Otwarte okno")
    axes[0].scatter(noise_pos[0], 0.1, marker='x', color='red', s=100, label="Kierunek hałasu")
    axes[0].legend(loc="upper right")

    # Mapa różnicowa (Redukcja w dB)
    im2 = axes[1].contourf(GX, GY, Z_diff, levels=30, cmap='coolwarm', vmin=-15, vmax=15)
    axes[1].set_title(f"Redukcja hałasu (ANC WŁĄCZONE) - Zysk [dB]\nGlobalne pochłonięcie energii: {energy_reduction_pct:.1f}%")
    axes[1].set_xlabel("Szerokość pokoju [m]")
    axes[1].set_ylabel("Długość pokoju [m]")
    fig.colorbar(im2, ax=axes[1], label="Różnica SPL [dB] (ujemna = ciszej)")
    
    # Rysuj głośniczki ANC
    axes[1].plot([1.5, 3.5], [0, 0], 'k-', linewidth=4, label="Okno")
    axes[1].scatter(speaker_positions[0, :], speaker_positions[1, :], color='green', marker='o', s=60, label="Jednostki ANC (głośniki)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    output_path = f"/home/przemyslaw/dev/Acoustics/anti_noise_window_result_{n_units}.jpg"
    plt.savefig(output_path, dpi=200)
    print(f"Wykres zapisano do: {output_path}")


if __name__ == "__main__":
    for n in range(1, 10):
        simulate_anti_noise_window(n)
