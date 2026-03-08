"""
Example: Simulating and Optimizing a 2-stage FH Generator with Single-Focus Geometry
=====================================================================================
Setup (from paper):
    1027 nm, 215 fs, 100 nJ pulses
    Stage 1: BIBO crystal (BiBO rot 47deg), SHG: 1027 nm -> 514 nm
    Stage 2: BBO crystal,                  SHG: 514 nm  -> 257 nm (4th harmonic)
    Both crystals placed inside a single focus (single Rayleigh range ~ 0.5mm each)

Goal:
    Prove the paper's claim that single-focus geometry achieves competitive FH efficiency
    with ultrafast (215 fs) pulses -- previously only shown for ns pulses.
    
    We do a 2D sweep over (BIBO thickness, BBO thickness) and compute FH conversion
    efficiency at each point, using LWE's CLI via SimulationRunner.

Usage:
    Make sure CLI is compiled and available at CLI_PATH below.
    Run: python3 example_fh_single_focus.py
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
import LightwaveExplorer as lwe

# =============================================================================
# Configuration - adjust these to match your system
# =============================================================================

CLI_PATH = "./build_cli/LightwaveExplorer"   # Path to compiled LWE CLI
WORK_DIR = "/tmp/lwe_fh_sweep"               # Temp working directory
OUTPUT_DIR = "/mnt/d/LWE_FH_Data"           # Where to store .zip results (on D: via WSL)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Laser base parameters (from paper: 1027 nm, 215 fs, ~100 nJ, 1 MHz rep)
WAVELENGTH_NM = 1027.0
PULSE_ENERGY_J = 100e-9       # 100 nJ
PULSE_DURATION_S = 215e-15    # 215 fs
BEAM_WAIST_M = 30e-6          # 30 µm beam waist (typical for 100mm focal lens)

# Phase-matching angles (starting points; ideally from Sellmeier calculation)
# BIBO (BiBO rot 47deg, crystal index 4): Type II SHG at 1027nm, theta ~ 11.4 deg
# BBO  (crystal index 1):               Type I  SHG at 514nm,  theta ~ 50.4 deg
BIBO_THETA_DEG = 11.4    # You can refine this
BBO_THETA_DEG  = 50.4    # You can refine this

# Crystal thickness sweep range (in metres)
BIBO_THICKNESSES = np.linspace(0.2e-3, 1.5e-3, 7)  # 0.2mm to 1.5mm
BBO_THICKNESSES  = np.linspace(0.2e-3, 1.5e-3, 7)  # 0.2mm to 1.5mm

# Frequency bounds for extracting FH signal (257nm = 1167 THz)
FH_FREQ_CENTER_THz = 1167e12   # 257 nm
FH_FREQ_WINDOW_THz = 50e12    # ±50 THz window around FH

# =============================================================================
# 1. Define the Sequence Generator
# =============================================================================

def make_sequence(bibo_thickness_m: float, bbo_thickness_m: float) -> str:
    """
    Build a LWE sequence string for 2-stage SHG in single-focus geometry.
    
    LWE sequence elements used:
      crystal(idx, theta_deg, phi_deg)   - switch to crystal material
      propagate(thickness_m)             - propagate through the crystal
      defaultMaterial()                  - switch back to free space
      save(1)                            - save the result at this point
    """
    # Stage 1: BIBO = crystal index 4 (BiBO rot 47deg)
    # Stage 2: BBO  = crystal index 1
    return (
        f"crystal(4, {BIBO_THETA_DEG:.4f}, 0) "
        f"propagate({bibo_thickness_m:.6e}) "
        f"defaultMaterial() "
        f"crystal(1, {BBO_THETA_DEG:.4f}, 0) "
        f"propagate({bbo_thickness_m:.6e}) "
        f"defaultMaterial() "
        f"save(1)"
    )

# =============================================================================
# 2. Define the Metric: FH conversion efficiency
# =============================================================================

def fh_efficiency(result: lwe.lightwaveExplorerResult) -> float:
    """
    Compute the fraction of output energy in the 4th harmonic band (around 257nm).
    
    Uses the spectrumTotal array (from _spectrum.dat), which is already in
    units of spectral intensity vs. frequency index. We integrate over the
    frequency range near the FH.
    """
    freq = result.frequencyVectorSpectrum   # 1D array of frequencies in Hz
    spectrum = result.spectrumTotal         # 1D or 2D array; last save point
    
    # If batch result, pick the last simulation
    if spectrum.ndim > 1:
        spectrum = spectrum[-1]
    
    # Mask out only the FH frequency window
    fh_mask = np.abs(freq - FH_FREQ_CENTER_THz) < FH_FREQ_WINDOW_THz
    
    total_power = np.trapz(spectrum, freq)
    fh_power    = np.trapz(spectrum[fh_mask], freq[fh_mask])
    
    if total_power <= 0:
        return 0.0
    return fh_power / total_power

# =============================================================================
# 3. Run the 2D Sweep
# =============================================================================

print("Initializing SimulationRunner...")
runner = lwe.SimulationRunner(cli_path=CLI_PATH, work_dir=WORK_DIR)

# Set the base laser parameters (these stay constant across the sweep)
runner.set_params(
    frequency1        = 3e8 / (WAVELENGTH_NM * 1e-9),   # Convert nm to Hz
    pulse_energy1     = PULSE_ENERGY_J,
    bandwidth1        = 0.4415 / PULSE_DURATION_S,       # TBP for sech2 pulse
    beamwaist1        = BEAM_WAIST_M,
    symmetry_type     = 1,        # 1 = cylindrical symmetry (fast!)
    time_span         = 3e-12,    # 3 ps window
    dt                = 2e-15,    # 2 fs time step
    grid_width        = 150e-6,   # 150 µm radial grid
    dx                = 3e-6,     # 3 µm spatial step
)

print(f"Running {len(BIBO_THICKNESSES) * len(BBO_THICKNESSES)} simulations...")
print(f"BIBO sweep: {BIBO_THICKNESSES*1e3} mm")
print(f"BBO  sweep: {BBO_THICKNESSES*1e3} mm\n")

efficiency_map = np.zeros((len(BIBO_THICKNESSES), len(BBO_THICKNESSES)))

for i, bibo_t in enumerate(BIBO_THICKNESSES):
    for j, bbo_t in enumerate(BBO_THICKNESSES):
        seq = make_sequence(bibo_t, bbo_t)
        runner.set_params(sequence=seq)
        
        label = f"bibo{bibo_t*1e3:.2f}mm_bbo{bbo_t*1e3:.2f}mm"
        output_path = os.path.join(OUTPUT_DIR, label)
        
        try:
            runner.run(output_name=output_path, verbose=False, load_field=False)
            if runner.result:
                eff = fh_efficiency(runner.result)
                efficiency_map[i, j] = eff
                print(f"  BIBO={bibo_t*1e3:.2f}mm, BBO={bbo_t*1e3:.2f}mm -> FH efficiency: {eff*100:.2f}%")
            else:
                print(f"  BIBO={bibo_t*1e3:.2f}mm, BBO={bbo_t*1e3:.2f}mm -> FAILED")
        except Exception as e:
            print(f"  BIBO={bibo_t*1e3:.2f}mm, BBO={bbo_t*1e3:.2f}mm -> ERROR: {e}")

# =============================================================================
# 4. Find Optimal and Plot
# =============================================================================

best_idx = np.unravel_index(np.argmax(efficiency_map), efficiency_map.shape)
best_bibo = BIBO_THICKNESSES[best_idx[0]]
best_bbo  = BBO_THICKNESSES[best_idx[1]]
best_eff  = efficiency_map[best_idx]

print(f"\n{'='*50}")
print(f"Optimal Configuration:")
print(f"  BIBO thickness: {best_bibo*1e3:.2f} mm")
print(f"  BBO  thickness: {best_bbo*1e3:.2f} mm")
print(f"  FH efficiency:  {best_eff*100:.2f}%")
print(f"{'='*50}")

# Plot the efficiency map
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    efficiency_map.T * 100,
    origin="lower",
    extent=[BIBO_THICKNESSES[0]*1e3, BIBO_THICKNESSES[-1]*1e3,
            BBO_THICKNESSES[0]*1e3,  BBO_THICKNESSES[-1]*1e3],
    aspect="auto", cmap="plasma"
)
plt.colorbar(im, ax=ax, label="FH Conversion Efficiency (%)")
ax.set_xlabel("BIBO Thickness (mm)")
ax.set_ylabel("BBO Thickness (mm)")
ax.set_title("4th Harmonic Conversion Efficiency\n(Single-Focus Geometry, 1027nm → 257nm, 215fs)")
ax.plot(best_bibo*1e3, best_bbo*1e3, "w*", markersize=15, label=f"Max: {best_eff*100:.1f}%")
ax.legend()
plt.tight_layout()
plt.savefig("fh_efficiency_map.png", dpi=150)
plt.show()
print("Plot saved to fh_efficiency_map.png")
