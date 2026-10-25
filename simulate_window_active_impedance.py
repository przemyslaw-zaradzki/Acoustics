"""
Symulacja wpływu sterowanych impedancji akustycznych w otworze okiennym
na pole akustyczne w pomieszczeniu (Pochłaniacz Elektroakustyczny / Active Impedance Boundary).

W odróżnieniu od klasycznego ANC:
- Jednostki nie emitują fali w dal jako autonomiczne źródła ciśnienia.
- Każda jednostka wymusza lokalną admitancję Y = u / p ≈ 1 / (rho * c) (dopasowanie impedancji, alpha -> 1).
- Wydatek objętościowy q_i = u_i * S_i = (p_inc / Z_target) * (W_window / N), dzięki czemu
  wraz ze wzrostem N aperturę pokrywamy coraz gęściej, a całkowita pochłonięta moc dąży
  stabilnie do wartości asymptotycznej bez efektu przeładowania energią (over-driving).
"""

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra


def simulate_active_impedance(n_units=4, freq_noise=150.0, save_plot=True):
    fs = 8000
    c = 343.0
    rho0 = 1.21  # Gęstość powietrza [kg/m^3]
    z0 = rho0 * c  # Właściwa impedancja falowa powietrza (~415 Pa*s/m)
    
    # ---------------------------------------------------------
    # 1. Geometria pokoju i okna
    # ---------------------------------------------------------
    room_dim = [5.0, 4.0]
    absorption = 0.15
    max_order = 3

    # Okno na dolnej ścianie y=0
    window_x_start = 1.7
    window_x_end = 3.3
    window_width = window_x_end - window_x_start  # 1.6 m
    window_y = 0.05

    # Zewnętrzne źródło hałasu
    noise_pos = [2.5, -2.0]

    # Pozycje N_units jednostek sterowanej impedancji w otworze okna
    # Każda jednostka odpowiada za segment apertury o szerokości dx = window_width / n_units
    dx = window_width / n_units
    # Środki poszczególnych segmentów
    speaker_x = window_x_start + (np.arange(n_units) + 0.5) * dx
    speaker_positions = np.zeros((2, n_units))
    speaker_positions[0, :] = speaker_x
    speaker_positions[1, :] = window_y

    # Siatka ewaluacyjna w pokoju
    grid_x = np.linspace(0.5, 4.5, 30)
    grid_y = np.linspace(0.5, 3.5, 25)
    GX, GY = np.meshgrid(grid_x, grid_y)
    eval_mics = np.vstack([GX.ravel(), GY.ravel()])

    # ---------------------------------------------------------
    # 2. Obliczenia falowe (dziedzina częstotliwości)
    # ---------------------------------------------------------
    omega = 2 * np.pi * freq_noise
    k = omega / c

    # Fala padająca na poszczególne jednostki w oknie (propagacja z zewnątrz)
    dist_noise_to_units = np.linalg.norm(speaker_positions.T - noise_pos, axis=1)
    # Ciśnienie fali padającej ze źródła zewnętrznego w punktach jednostek
    P_inc = (1.0 / dist_noise_to_units) * np.exp(-1j * k * dist_noise_to_units)

    # ---------------------------------------------------------
    # 3. Model propagacji wewnątrz pokoju (pyroomacoustics)
    # ---------------------------------------------------------
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption), max_order=max_order)

    # Dodajemy źródła wtórne (jednostki impedancyjne na granicy pokoju)
    for i in range(n_units):
        room.add_source(speaker_positions[:, i])

    # Hałas wnikający przez okno do pokoju (całkowity strumień fali pierwotnej)
    room.add_source([2.5, window_y])
    noise_virtual_idx = n_units

    room.add_microphone_array(pra.MicrophoneArray(eval_mics, fs=fs))
    room.compute_rir()

    n_eval_mics = eval_mics.shape[1]
    H_units = np.zeros((n_eval_mics, n_units), dtype=complex)
    H_noise = np.zeros(n_eval_mics, dtype=complex)

    fft_len = 4096
    freq_bins = np.fft.rfftfreq(fft_len, 1.0 / fs)
    target_bin = np.argmin(np.abs(freq_bins - freq_noise))

    for m in range(n_eval_mics):
        for s in range(n_units):
            H_fft = np.fft.rfft(room.rir[m][s], n=fft_len)
            H_units[m, s] = H_fft[target_bin]

        H_noise_fft = np.fft.rfft(room.rir[m][noise_virtual_idx], n=fft_len)
        H_noise[m] = H_noise_fft[target_bin]

    # ---------------------------------------------------------
    # 4. Sterowanie impedancyjne (Electroacoustic Absorber)
    # ---------------------------------------------------------
    # W układzie dopasowania impedancji:
    # Membrana o powierzchni dS = dx * h_unit generuje przepływ o przeciwnym znaku
    # u_i = - P_inc / Z_target, gdzie Z_target ≈ z0 (impedancja falowa powietrza).
    #
    # Kluczowe: siła/amplituda źródła dyskretnego jest skalowana przez szerokość dx segmentu!
    # Przy n_units -> inf, suma q_i zbiega do całki powierzchniowej.
    # Wprowadzamy współczynnik dopasowania (zysk pochłaniania):
    absorption_gain = 0.90  # 90% idealnego pochłonięcia
    # Normalizacja względem szerokości okna (dla zachowania stałej energii wlotu)
    unit_weight = (dx / window_width) * absorption_gain

    # Odpowiedź jednostek: przeciwfaza i proporcjonalność do lokalnego ciśnienia oraz dx
    speaker_weights = - unit_weight * P_inc

    # Pole ciśnienia
    P_off = H_noise * 1.0
    P_absorber = np.sum(H_units * speaker_weights[np.newaxis, :], axis=1)
    P_on = P_off + P_absorber

    # Energia akustyczna w pokoju
    energy_off = np.mean(np.abs(P_off) ** 2)
    energy_on = np.mean(np.abs(P_on) ** 2)
    energy_reduction_pct = (1.0 - energy_on / energy_off) * 100
    global_reduction_db = 10 * np.log10(energy_on / energy_off)

    print(f"N = {n_units:2d} | Energia OFF: {energy_off:.4e} | Energia ON: {energy_on:.4e} | "
          f"Redukcja: {energy_reduction_pct:6.2f}% ({global_reduction_db:+5.2f} dB)")

    if save_plot:
        spl_off = 10 * np.log10(np.abs(P_off) ** 2 + 1e-12)
        spl_on = 10 * np.log10(np.abs(P_on) ** 2 + 1e-12)
        diff_spl = spl_on - spl_off

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im1 = axes[0].contourf(GX, GY, spl_off.reshape(GX.shape), levels=30, cmap='inferno')
        axes[0].set_title(f"Rozkład ciśnienia BEZ aktywnych jednostek ({freq_noise:.0f} Hz)")
        axes[0].set_xlabel("Szerokość pokoju [m]")
        axes[0].set_ylabel("Długość pokoju [m]")
        fig.colorbar(im1, ax=axes[0], label="Względny SPL [dB]")
        axes[0].plot([window_x_start, window_x_end], [0, 0], 'c-', linewidth=6, label="Okno")
        axes[0].legend(loc="upper right")

        im2 = axes[1].contourf(GX, GY, diff_spl.reshape(GX.shape), levels=30, cmap='coolwarm', vmin=-10, vmax=5)
        axes[1].set_title(f"Redukcja ciśnienia ze sterowaną impedancją (N = {n_units})\n"
                          f"Globalna redukcja energii: {energy_reduction_pct:.2f}% ({global_reduction_db:.2f} dB)")
        axes[1].set_xlabel("Szerokość pokoju [m]")
        axes[1].set_ylabel("Długość pokoju [m]")
        fig.colorbar(im2, ax=axes[1], label="Różnica SPL [dB] (ujemna = ciszej)")
        
        axes[1].plot([window_x_start, window_x_end], [0, 0], 'k-', linewidth=4, label="Okno")
        axes[1].scatter(speaker_positions[0, :], speaker_positions[1, :],
                        color='green', marker='s', s=60, label=f"Jednostki impedancyjne (N={n_units})")
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        out_name = f"/home/przemyslaw/dev/Acoustics/active_impedance_result_N{n_units}.png"
        plt.savefig(out_name, dpi=200)
        plt.close()

    return energy_reduction_pct, global_reduction_db


if __name__ == "__main__":
    print("=" * 70)
    print("SYMULACJA STEROWANYCH IMPEDANCJI AKUSTYCZNYCH DLA RÓŻNYCH N:")
    print("=" * 70)
    
    n_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    reductions = []
    
    for n in n_list:
        red_pct, red_db = simulate_active_impedance(n_units=n, freq_noise=150.0, save_plot=(n in [1, 4, 8]))
        reductions.append(red_pct)

    # Wykres zbieżności redukcji w funkcji liczby jednostek N
    plt.figure(figsize=(8, 4.5))
    plt.plot(n_list, reductions, 'o-', color='navy', linewidth=2, markersize=7)
    plt.axhline(y=max(reductions), color='r', linestyle='--', alpha=0.6, label=f"Maksimum: {max(reductions):.1f}%")
    plt.title("Wpływ liczby jednostek sterowanej impedancji (N) na redukcję energii")
    plt.xlabel("Liczba jednostek w oknie (N)")
    plt.ylabel("Pochłonięta energia w pokoju [%]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/przemyslaw/dev/Acoustics/active_impedance_convergence.png", dpi=200)
    plt.close()
    print("=" * 70)
    print("Wykres zbieżności zapisano do: /home/przemyslaw/dev/Acoustics/active_impedance_convergence.png")

