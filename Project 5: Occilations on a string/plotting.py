import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

"""
File Header:

Author: Nels Buhrley
Date: 2024-06-15

Description:
This script loads the results of a string oscillation simulation from a .npz file,and creates a figure with
three components:
    1. A plot of the initial string displacement at time t = 0.
    2. A plot of the mean power spectrum across all time steps, showing the dominant frequencies of oscillation.
    3. An animation of the string's displacement over time, illustrating the propagation of waves along the string.
        The animation is saved as an MP4 file for later viewing.
The .npz file is expected to contain the following arrays:
    - "positions": A 2D array of shape (timeSteps, segments) containing the displacement of each segment of
        the string at each time step.
    - "frequencies": A 1D array containing the frequency bins corresponding to the power spectrum.
    - "mean_power_spectrum": A 1D array containing the mean power at each frequency bin, averaged over all time steps.
    - "parameters": A 1D array containing the simulation parameters: [length, waveSpeed, stiffness, r, timeStep].


"""

# ── Load data ────────────────────────────────────────────────────────────────
npz_path = os.path.join(os.path.dirname(__file__), "string_oscillations.npz")
data = np.load(npz_path)

positions      = data["positions"]        # shape: (timeSteps, segments)
frequencies    = data["frequencies"]      # shape: (numBins,)
mean_power     = data["mean_power_spectrum"]  # shape: (numBins,)
parameters     = data["parameters"]       # shape: (5,)  [length, waveSpeed, stiffness, r, timeStep]
dt = parameters[4]

time_steps, segments = positions.shape
print(f"Positions shape : {positions.shape}")
print(f"Frequency bins  : {frequencies.shape[0]}  (0 – {frequencies[-1]:.1f} Hz)")

length = 1.0
x = np.linspace(0, length, segments)

# ── Figure layout ─────────────────────────────────────────────────────────────
#   Row 0: initial state (left)  |  power spectrum (right)
#   Row 1: animation             (full width)
fig = plt.figure(figsize=(14, 9))
gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], hspace=0.45, wspace=0.35)

ax_init   = fig.add_subplot(gs[0, 0])
ax_spec   = fig.add_subplot(gs[0, 1])
ax_anim   = fig.add_subplot(gs[1, :])

# ── 1. Initial state ──────────────────────────────────────────────────────────
ax_init.plot(x, positions[0], color="steelblue", linewidth=2)
ax_init.set_xlabel("Position along string (m)", fontsize=11)
ax_init.set_ylabel("Displacement", fontsize=11)
ax_init.set_xlim(0, length)
ax_init.set_title("Initial String Displacement  (t = 0)", fontsize=12, fontweight="bold")
ax_init.grid(True, alpha=0.3)

# ── 2. Mean power spectrum ────────────────────────────────────────────────────
# Clip to a sensible frequency range (ignore DC and very high bins with ~0 power)
f_max_plot = frequencies[np.argmax(mean_power) * 30] if np.argmax(mean_power) > 0 else frequencies[-1]
f_max_plot = min(f_max_plot, frequencies[-1])
mask = frequencies <= f_max_plot

ax_spec.semilogy(frequencies[mask], mean_power[mask], color="darkorange", linewidth=1.5)
ax_spec.set_xlabel("Frequency (Hz)", fontsize=11)
ax_spec.set_ylabel("Mean Power  (log scale)", fontsize=11)
ax_spec.set_title("Mean Power Spectrum", fontsize=12, fontweight="bold")
ax_spec.grid(True, alpha=0.3, which="both")
ax_spec.set_xlim(0, f_max_plot)

# ── 3. Animation ──────────────────────────────────────────────────────────────
fps        = 5
duration_s = 180
n_frames   = fps * duration_s
interval_ms = 1000 / fps
max_time_index = min(100000, time_steps - 1)

output_mp4        = os.path.join(os.path.dirname(__file__), "string_oscillations.mp4")

time_indices = np.linspace(0, max_time_index, n_frames).astype(int)
n_frames     = len(time_indices)

y_min = positions[time_indices].min()
y_max = positions[time_indices].max()
y_pad = (y_max - y_min) * 0.1 or 0.01

anim_line, = ax_anim.plot(x, positions[time_indices[0]], color="steelblue", linewidth=2)
ax_anim.set_xlabel("Position along string (m)", fontsize=11)
ax_anim.set_ylabel("Displacement", fontsize=11)
ax_anim.set_xlim(0, length)
ax_anim.set_ylim(y_min - y_pad, y_max + y_pad)
ax_anim.grid(True, alpha=0.3)
anim_title = ax_anim.set_title("", fontsize=12, fontweight="bold")

def update(frame_idx):
    t = time_indices[frame_idx]
    anim_line.set_ydata(positions[t])
    anim_title.set_text(
        f"String Propagation  |  time: {t * dt:.5f}s  |  frame: {frame_idx + 1}/{n_frames}"
    )
    return anim_line, anim_title

ani = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=1000 / fps, blit=False
)

plt.suptitle("String Oscillations", fontsize=15, fontweight="bold", y=1.01)

print(f"Saving animation: {n_frames} frames at {fps} fps → {output_mp4}")
writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
ani.save(output_mp4, writer=writer, dpi=120,
         progress_callback=lambda i, n: print(f"  frame {i}/{n}", end="\r"))
print(f"\nSaved → {output_mp4}")
plt.show()
