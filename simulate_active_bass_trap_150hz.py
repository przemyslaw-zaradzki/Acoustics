"""
Symulacja wpływu aktywnych pułapek basowych (AVAA / ANC w narożnikach)
dla częstotliwości 150 Hz (analogicznie do simulate_window_anc.py).

Porównanie:
1. Niskie częstotliwości (34 Hz) vs wyższy bas (150 Hz).
2. Lokalny feedback w narożnikach (Local Corner Cancellation).
3. Optymalne globalne sterowanie ANC (Global Least Squares).
"""

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra


def simulate_bass_trap_150hz(n_corners=4, freq_target=150.0):
    """
    Symulacja aktywnej pułapki basowej dla częstotliwości 150 Hz w pokoju 3D.
    """
    fs = 8000
    c = 343.0
    wavelength = c / freq_target  # Długość fali dla 150 Hz wynosi ok. 2.29 m
    
    # ---------------------------------------------------------
    # 1. Geometria pokoju: 5.0m x 4.0m x 2.8m
    # ---------------------------------------------------------
    room_dim = [5.0, 4.0, 2.8]
    absorption = 0.12  # Typowe pochłanianie dla 150 Hz w umeblowanym pokoju
    max_order = 10
    
    # Źródło hałasu/dudnienia 150 Hz w pokoju (np. maszyna, subwoofer, rezonans instalacji)
    noise_source_pos = [0.8, 0.7, 0.5]
    
    # ---------------------------------------------------------
    # 2. Pozycje aktywnych pułapek (głośniki i mikrofony błędu)
    # ---------------------------------------------------------
    corner_speakers = np.array([
        [0.20, 0.20, 0.30],                    # Narożnik 1: Lewy-przód
        [room_dim[0] - 0.20, 0.20, 0.30],      # Narożnik 2: Prawy-przód
        [room_dim[0] - 0.20, room_dim[1] - 0.20, 0.30],  # Narożnik 3: Prawy-tył
        [0.20, room_dim[1] - 0.20, 0.30],      # Narożnik 4: Lewy-tył
    ])[:n_corners].T  # (3, n_corners)
    
    # Mikrofony kolokowane przy głośnikach pułapek (odsunięte o 8 cm)
    mic_offsets = np.array([
        [0.08, 0.08, 0.0],
        [-0.08, 0.08, 0.0],
        [-0.08, -0.08, 0.0],
        [0.08, -0.08, 0.0],
    ])[:n_corners].T
    corner_mics = corner_speakers + mic_offsets
    
    # Siatka mikrofonów pomiarowych do oceny całego pokoju (wysokość z=1.2 m)
    grid_x = np.linspace(0.3, room_dim[0] - 0.3, 30)
    grid_y = np.linspace(0.3, room_dim[1] - 0.3, 24)
    GX, GY = np.meshgrid(grid_x, grid_y)
    eval_mics = np.vstack([GX.ravel(), GY.ravel(), np.full(GX.size, 1.2)])
    
    all_mics = np.hstack([corner_mics, eval_mics])
    
    # ---------------------------------------------------------
    # 3. Modelowanie akustyczne w pyroomacoustics
    # ---------------------------------------------------------
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption), max_order=max_order)
    
    # Dodajemy źródło hałasu
    room.add_source(noise_source_pos)
    noise_idx = 0
    
    # Dodajemy głośniki pułapek narożnych
    for i in range(n_corners):
        room.add_source(corner_speakers[:, i])
    trap_start_idx = 1
    
    room.add_microphone_array(pra.MicrophoneArray(all_mics, fs=fs))
    room.compute_rir()
    
    # ---------------------------------------------------------
    # 4. Obliczenia w dziedzinie częstotliwości dla 150 Hz
    # ---------------------------------------------------------
    fft_len = 8192
    freq_bins = np.fft.rfftfreq(fft_len, 1.0 / fs)
    target_bin = np.argmin(np.abs(freq_bins - freq_target))
    actual_freq = freq_bins[target_bin]
    
    # A. Transmitancje do mikrofonów w narożnikach
    P_noise_mics = np.zeros(n_corners, dtype=complex)
    for i in range(n_corners):
        rir = room.rir[i][noise_idx]
        P_noise_mics[i] = np.fft.rfft(rir, n=fft_len)[target_bin]
        
    Z_traps = np.zeros((n_corners, n_corners), dtype=complex)
    for m in range(n_corners):
        for s in range(n_corners):
            rir = room.rir[m][trap_start_idx + s]
            Z_traps[m, s] = np.fft.rfft(rir, n=fft_len)[target_bin]
            
    # B. Transmitancje do punktów w całym pomieszczeniu
    n_eval = eval_mics.shape[1]
    H_noise_eval = np.zeros(n_eval, dtype=complex)
    H_traps_eval = np.zeros((n_eval, n_corners), dtype=complex)
    
    for m_idx in range(n_eval):
        mic_room_idx = n_corners + m_idx
        rir_n = room.rir[mic_room_idx][noise_idx]
        H_noise_eval[m_idx] = np.fft.rfft(rir_n, n=fft_len)[target_bin]
        
        for s in range(n_corners):
            rir_t = room.rir[mic_room_idx][trap_start_idx + s]
            H_traps_eval[m_idx, s] = np.fft.rfft(rir_t, n=fft_len)[target_bin]
            
    P_off = H_noise_eval
    energy_off = np.mean(np.abs(P_off) ** 2)
    
    # ---------------------------------------------------------
    # 5. Dwie strategie sterowania pułapkami basowymi przy 150 Hz
    # ---------------------------------------------------------
    
    # Strategia 1: Zdecentralizowany lokalny feedback w narożnikach (Pure Local Feedback)
    # Dąży do wyzerowania ciśnienia w samym narożniku
    reg_local = 5e-2 * np.eye(n_corners) * np.max(np.abs(Z_traps))
    weights_local = - np.linalg.solve(Z_traps + reg_local, P_noise_mics)
    P_on_local = P_off + H_traps_eval @ weights_local
    energy_local = np.mean(np.abs(P_on_local) ** 2)
    red_db_local = 10 * np.log10(energy_local / energy_off)
    
    # Strategia 2: Globalnie zoptymalizowane sterowanie ANC (Optimal Multichannel ANC)
    # Minimalizuje sumaryczną energię w przestrzeni odsłuchowej
    weights_opt, _, _, _ = np.linalg.lstsq(H_traps_eval, -P_off, rcond=1e-3)
    P_on_opt = P_off + H_traps_eval @ weights_opt
    energy_opt = np.mean(np.abs(P_on_opt) ** 2)
    red_db_opt = 10 * np.log10(energy_opt / energy_off)
    pct_opt = (1.0 - energy_opt / energy_off) * 100
    
    log_message = f"""============================================================
WYNIKI SYMULACJI AKTYWNEJ PUŁAPKI DLA HAŁASU {freq_target:.1f} Hz (λ = {wavelength:.2f} m):
Liczba narożników: {n_corners}
Średnia energia hałasu BEZ pułapek: {energy_off:.5e}

1. LOKALNY FEEDBACK W NAROŻNIKACH (czyste zerowanie narożników):
   - Średnia energia w pokoju:      {energy_local:.5e}
   - Zmiana poziomu dźwięku:        {red_db_local:+.2f} dB (efekt spillover/podbić w środku)

2. GLOBALNIE ZOPTYMALIZOWANE ANC (4 źródła narożne):
   - Średnia energia w pokoju:      {energy_opt:.5e}
   - Redukcja energii akustycznej:  {pct_opt:.2f}%
   - Średnia redukcja poziomu SPL:  {red_db_opt:.2f} dB
============================================================"""
    print(log_message)
    
    with open(f"active_bass_trap_150hz_{n_corners}corners.log", "w", encoding="utf-8") as f:
        f.write(log_message + "\n")
        
    # ---------------------------------------------------------
    # 6. Wizualizacja: Porównanie map ciśnienia
    # ---------------------------------------------------------
    spl_off = 10 * np.log10(np.abs(P_off) ** 2 + 1e-12)
    spl_local = 10 * np.log10(np.abs(P_on_local) ** 2 + 1e-12)
    spl_opt = 10 * np.log10(np.abs(P_on_opt) ** 2 + 1e-12)
    
    diff_opt = spl_opt - spl_off
    ref_max = np.max(spl_off)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Wykres 1: Pole przed tłumieniem
    im1 = axes[0].contourf(GX, GY, (spl_off - ref_max).reshape(GX.shape), levels=35, cmap='inferno', vmin=-25, vmax=0)
    axes[0].scatter(noise_source_pos[0], noise_source_pos[1], color='cyan', marker='s', s=120, edgecolors='black', label="Źródło hałasu 150 Hz")
    axes[0].set_title(f"Rozkład hałasu 150 Hz BEZ pułapek\n(λ = {wavelength:.2f} m)")
    axes[0].set_xlabel("Szerokość pokoju X [m]")
    axes[0].set_ylabel("Głębokość pokoju Y [m]")
    axes[0].legend(loc="upper right")
    fig.colorbar(im1, ax=axes[0], label="Względny SPL [dB]")
    
    # Wykres 2: Pole po włączeniu zoptymalizowanego ANC z narożników
    im2 = axes[1].contourf(GX, GY, (spl_opt - ref_max).reshape(GX.shape), levels=35, cmap='inferno', vmin=-25, vmax=0)
    axes[1].scatter(noise_source_pos[0], noise_source_pos[1], color='cyan', marker='s', s=120, edgecolors='black', label="Źródło hałasu")
    axes[1].scatter(corner_speakers[0, :], corner_speakers[1, :], color='lime', marker='o', s=100, edgecolors='black', label="Głośniki ANC w rogach")
    axes[1].set_title(f"Z pułapkami narożnymi ANC (150 Hz)\nRedukcja globalna: {red_db_opt:.1f} dB")
    axes[1].set_xlabel("Szerokość pokoju X [m]")
    axes[1].set_ylabel("Głębokość pokoju Y [m]")
    axes[1].legend(loc="upper right")
    fig.colorbar(im2, ax=axes[1], label="Względny SPL [dB]")
    
    # Wykres 3: Różnica (Zysk w dB)
    im3 = axes[2].contourf(GX, GY, diff_opt.reshape(GX.shape), levels=35, cmap='coolwarm', vmin=-20, vmax=5)
    axes[2].scatter(corner_speakers[0, :], corner_speakers[1, :], color='green', marker='o', s=100, edgecolors='black', label="Głośniki ANC")
    axes[2].set_title(f"Redukcja ciśnienia w dB (Optymalne ANC)\nPochłonięta energia: {pct_opt:.1f}%")
    axes[2].set_xlabel("Szerokość pokoju X [m]")
    axes[2].set_ylabel("Głębokość pokoju Y [m]")
    axes[2].legend(loc="upper right")
    fig.colorbar(im3, ax=axes[2], label="Różnica SPL [dB] (ujemna = ciszej)")
    
    plt.tight_layout()
    output_path = f"/home/przemyslaw/dev/Acoustics/active_bass_trap_150hz_{n_corners}corners.jpg"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wykres zapisano do: {output_path}")


if __name__ == "__main__":
    print("--- Symulacja aktywnej pułapki dla 150 Hz (2 narożniki) ---")
    simulate_bass_trap_150hz(n_corners=2, freq_target=150.0)
    
    print("\n--- Symulacja aktywnej pułapki dla 150 Hz (4 narożniki) ---")
    simulate_bass_trap_150hz(n_corners=4, freq_target=150.0)

