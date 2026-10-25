"""
Symulacja aktywnej pułapki basowej (Active Bass Trap / AVAA) w narożnikach pokoju
z wykorzystaniem biblioteki pyroomacoustics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra


def simulate_active_bass_trap(n_corners=4, target_mode=(1, 0, 0)):
    """
    Symulacja aktywnej pułapki basowej w narożnikach pokoju trójwymiarowego.
    
    :param n_corners: Liczba narożników wyposażonych w aktywną pułapkę (1, 2 lub 4).
    :param target_mode: Indeksy modu własnego pokoju (nx, ny, nz), np. (1, 0, 0) dla modu osiowego wzdłuż długości.
    """
    fs = 8000  # Częstotliwość próbkowania [Hz]
    c = 343.0  # Prędkość dźwięku [m/s]
    
    # ---------------------------------------------------------
    # 1. Geometria pokoju odsłuchowego: 5.0m x 4.0m x 2.8m
    # ---------------------------------------------------------
    room_dim = [5.0, 4.0, 2.8]
    
    # Teoretyczna częstotliwość rezonansowa modu (nx, ny, nz)
    nx, ny, nz = target_mode
    freq_mode = (c / 2.0) * np.sqrt(
        (nx / room_dim[0]) ** 2 + (ny / room_dim[1]) ** 2 + (nz / room_dim[2]) ** 2
    )
    
    # Współczynnik pochłaniania dla niskich częstotliwości (ściany masywne, beton/cegła: niski współczynnik)
    absorption = 0.05
    max_order = 14  # Rząd odbić ISM
    
    # Pozycja subwoofera (źródło pierwotne generujące dudnienie/falę stojącą)
    subwoofer_pos = [0.6, 0.5, 0.4]
    
    # ---------------------------------------------------------
    # 2. Rozmieszczenie aktywnych pułapek basowych w narożnikach (AVAA)
    # W narożnikach pomieszczenia ciśnienie fali stojącej osiąga maksimum (strzałka ciśnienia)
    # ---------------------------------------------------------
    corner_speakers = [
        [0.20, 0.20, 0.30],                    # Narożnik 1: Lewy-przód
        [room_dim[0] - 0.20, 0.20, 0.30],      # Narożnik 2: Prawy-przód
        [room_dim[0] - 0.20, room_dim[1] - 0.20, 0.30],  # Narożnik 3: Prawy-tył
        [0.20, room_dim[1] - 0.20, 0.30],      # Narożnik 4: Lewy-tył
    ]
    
    active_corners = np.array(corner_speakers[:n_corners]).T  # Kształt: (3, n_corners)
    
    # Mikrofony błędu (kolokowane z głośnikami pułapek, odsunięte o 8 cm w głąb pokoju)
    mic_offsets = np.array([
        [0.08, 0.08, 0.0],
        [-0.08, 0.08, 0.0],
        [-0.08, -0.08, 0.0],
        [0.08, -0.08, 0.0],
    ])[:n_corners].T
    corner_mics = active_corners + mic_offsets
    
    # Siatka pomiarowa na wysokości uszu słuchacza (z = 1.2 m)
    grid_x = np.linspace(0.3, room_dim[0] - 0.3, 30)
    grid_y = np.linspace(0.3, room_dim[1] - 0.3, 24)
    GX, GY = np.meshgrid(grid_x, grid_y)
    eval_mics = np.vstack([GX.ravel(), GY.ravel(), np.full(GX.size, 1.2)])  # Kształt: (3, N_eval)
    
    # Połączona tablica mikrofonów: najpierw mikrofony pułapek, potem siatka pokoju
    all_mics = np.hstack([corner_mics, eval_mics])
    
    # ---------------------------------------------------------
    # 3. Modelowanie akustyczne w pyroomacoustics
    # ---------------------------------------------------------
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption), max_order=max_order)
    
    # Dodajemy źródło pierwotne (subwoofer)
    room.add_source(subwoofer_pos)
    subwoofer_idx = 0
    
    # Dodajemy głośniki wtórne pułapek basowych
    for i in range(n_corners):
        room.add_source(active_corners[:, i])
    trap_start_idx = 1
    
    # Dodajemy wszystkie mikrofony
    room.add_microphone_array(pra.MicrophoneArray(all_mics, fs=fs))
    room.compute_rir()
    
    # ---------------------------------------------------------
    # 4. Wyznaczenie transmitancji H(f) i sygnałów sterujących pułapek
    # ---------------------------------------------------------
    fft_len = 8192
    freq_bins = np.fft.rfftfreq(fft_len, 1.0 / fs)
    target_bin = np.argmin(np.abs(freq_bins - freq_mode))
    actual_freq = freq_bins[target_bin]
    
    # A. Ciśnienie z subwoofera dochodzące do mikrofonów w pułapkach
    P_sub_mics = np.zeros(n_corners, dtype=complex)
    for i in range(n_corners):
        rir = room.rir[i][subwoofer_idx]
        P_sub_mics[i] = np.fft.rfft(rir, n=fft_len)[target_bin]
        
    # B. Wzajemne transmitancje między głośnikami a mikrofonami pułapek
    Z_traps = np.zeros((n_corners, n_corners), dtype=complex)
    for m in range(n_corners):
        for s in range(n_corners):
            src_idx = trap_start_idx + s
            rir = room.rir[m][src_idx]
            Z_traps[m, s] = np.fft.rfft(rir, n=fft_len)[target_bin]
            
    # C. Sterowanie feedback: zerowanie ciśnienia akustycznego w narożniku
    # Warunek: P_sub_mics + Z_traps * Q = 0  =>  Q = - (Z_traps + reg)^(-1) * P_sub_mics
    reg = 1e-2 * np.eye(n_corners) * np.max(np.abs(Z_traps))
    trap_weights = - np.linalg.solve(Z_traps + reg, P_sub_mics)
    
    # ---------------------------------------------------------
    # 5. Obliczenie pola ciśnienia i redukcji energii w pokoju
    # ---------------------------------------------------------
    n_eval = eval_mics.shape[1]
    H_sub_eval = np.zeros(n_eval, dtype=complex)
    H_traps_eval = np.zeros((n_eval, n_corners), dtype=complex)
    
    for m_idx in range(n_eval):
        mic_room_idx = n_corners + m_idx
        # Transmitancja od subwoofera
        rir_sub = room.rir[mic_room_idx][subwoofer_idx]
        H_sub_eval[m_idx] = np.fft.rfft(rir_sub, n=fft_len)[target_bin]
        # Transmitancje od pułapek
        for s in range(n_corners):
            src_idx = trap_start_idx + s
            rir_trap = room.rir[mic_room_idx][src_idx]
            H_traps_eval[m_idx, s] = np.fft.rfft(rir_trap, n=fft_len)[target_bin]
            
    # Ciśnienie bez pułapek (sam subwoofer)
    P_off = H_sub_eval
    # Ciśnienie z aktywnymi pułapkami
    P_on = P_off + H_traps_eval @ trap_weights
    
    # Średnia gęstość energii w pomieszczeniu
    energy_off = np.mean(np.abs(P_off) ** 2)
    energy_on = np.mean(np.abs(P_on) ** 2)
    energy_reduction_pct = (1.0 - energy_on / energy_off) * 100
    global_reduction_db = 10 * np.log10(energy_on / energy_off)
    
    log_message = f"""============================================================
WYNIKI SYMULACJI AKTYWNEJ PUŁAPKI BASOWEJ (AVAA):
Analizowany mod pokoju: {target_mode} -> f = {actual_freq:.1f} Hz (teor. {freq_mode:.1f} Hz)
Liczba aktywnych pułapek w narożnikach: {n_corners}
Położenie subwoofera: {subwoofer_pos}
Średnia energia rezonansu bez pułapki: {energy_off:.5e}
Średnia energia rezonansu z pułapką:   {energy_on:.5e}
Procent zredukowanej energii modu:     {energy_reduction_pct:.2f}%
Średnia redukcja poziomu ciśnienia:    {global_reduction_db:.2f} dB
============================================================"""
    print(log_message)
    
    with open(f"active_bass_trap_result_{n_corners}corners.log", "w", encoding="utf-8") as f:
        f.write(log_message + "\n")
        
    # ---------------------------------------------------------
    # 6. Wizualizacja pola ciśnienia SPL [dB]
    # ---------------------------------------------------------
    spl_off = 10 * np.log10(np.abs(P_off) ** 2 + 1e-12)
    spl_on = 10 * np.log10(np.abs(P_on) ** 2 + 1e-12)
    diff_spl = spl_on - spl_off
    
    ref_max = np.max(spl_off)
    spl_off_norm = spl_off - ref_max
    spl_on_norm = spl_on - ref_max
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Pole ciśnienia przed załączeniem pułapek
    im1 = axes[0].contourf(GX, GY, spl_off_norm.reshape(GX.shape), levels=35, cmap='inferno', vmin=-25, vmax=0)
    axes[0].scatter(subwoofer_pos[0], subwoofer_pos[1], color='cyan', marker='s', s=120, edgecolors='black', label="Subwoofer")
    axes[0].set_title(f"Rezonans modu {target_mode} BEZ pułapek\n(f = {actual_freq:.1f} Hz)")
    axes[0].set_xlabel("Szerokość pokoju X [m]")
    axes[0].set_ylabel("Głębokość pokoju Y [m]")
    axes[0].legend(loc="upper right")
    fig.colorbar(im1, ax=axes[0], label="Względny SPL [dB]")
    
    # 2. Pole ciśnienia po włączeniu pułapek basowych
    im2 = axes[1].contourf(GX, GY, spl_on_norm.reshape(GX.shape), levels=35, cmap='inferno', vmin=-25, vmax=0)
    axes[1].scatter(subwoofer_pos[0], subwoofer_pos[1], color='cyan', marker='s', s=120, edgecolors='black', label="Subwoofer")
    axes[1].scatter(active_corners[0, :], active_corners[1, :], color='lime', marker='o', s=100, edgecolors='black', label="Aktywne pułapki (narożniki)")
    axes[1].set_title(f"Z AKTYWNYMI PUŁAPKAMI ({n_corners} narożniki)\nWygaszenie ciśnienia narożnego")
    axes[1].set_xlabel("Szerokość pokoju X [m]")
    axes[1].set_ylabel("Głębokość pokoju Y [m]")
    axes[1].legend(loc="upper right")
    fig.colorbar(im2, ax=axes[1], label="Względny SPL [dB]")
    
    # 3. Różnica tłumienia w dB
    im3 = axes[2].contourf(GX, GY, diff_spl.reshape(GX.shape), levels=35, cmap='coolwarm', vmin=-25, vmax=5)
    axes[2].scatter(active_corners[0, :], active_corners[1, :], color='green', marker='o', s=100, edgecolors='black', label="Aktywne pułapki")
    axes[2].set_title(f"Redukcja rezonansu w dB\nŚrednie pochłonięcie energii: {energy_reduction_pct:.1f}% ({global_reduction_db:.1f} dB)")
    axes[2].set_xlabel("Szerokość pokoju X [m]")
    axes[2].set_ylabel("Głębokość pokoju Y [m]")
    axes[2].legend(loc="upper right")
    fig.colorbar(im3, ax=axes[2], label="Różnica SPL [dB] (ujemna = wygaszenie)")
    
    plt.tight_layout()
    output_path = f"/home/przemyslaw/dev/Acoustics/active_bass_trap_{n_corners}corners_mode_{nx}_{ny}_{nz}.jpg"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wykres zapisano do: {output_path}")


if __name__ == "__main__":
    # Symulacja modu osiowego (1, 0, 0) o częstotliwości 34.3 Hz:
    # 1) Dla 2 pułapek w narożnikach przednich
    print("--- Symulacja: 2 pułapki w narożnikach przednich ---")
    simulate_active_bass_trap(n_corners=2, target_mode=(1, 0, 0))
    
    # 2) Dla 4 pułapek we wszystkich 4 narożnikach pokoju
    print("\n--- Symulacja: 4 pułapki we wszystkich narożnikach pokoju ---")
    simulate_active_bass_trap(n_corners=4, target_mode=(1, 0, 0))

