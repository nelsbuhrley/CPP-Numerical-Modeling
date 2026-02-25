import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

"""
File Header:

Author: Nels Buhrley
Date: 2024-06-15

Description:

"""

# ── Load data ────────────────────────────────────────────────────────────────
npz_path = os.path.join(os.path.dirname(__file__), "output/defusion_output.npz")
data = np.load(npz_path)
metadata = data['metadata']  # shape: (2,)
cube_size = metadata[0]
movement_radius = metadata[1]

# ── Create animation ─────────────────────────────────────────────────────────
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([], [], [], s=1)
ax.set_xlim(-cube_size/2, cube_size/2)
ax.set_ylim(-cube_size/2, cube_size/2)
ax.set_zlim(-cube_size/2, cube_size/2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Particle Diffusion Simulation')

all_points = data['points']  # shape: (steps, numPoints, 3)

# ── Animation controls ────────────────────────────────────────────────────────
FPS          = 30     # frames per second for display and saved video
STEP_SKIP    = 1      # number of data points skipped between frames
FINAL_INDEX  = None      # last data index to animate (None = use all data)

# Build frame indices from 0 to FINAL_INDEX (inclusive), stepping by STEP_SKIP
end = FINAL_INDEX if FINAL_INDEX is not None else all_points.shape[0] - 1
frame_indices = np.arange(0, end + 1, STEP_SKIP)

def update(frame_idx):
    points = all_points[frame_idx]  # shape: (numPoints, 3)
    scat._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
    ax.set_title(f'Particle Diffusion — step {frame_idx}')
    return scat,

ani = animation.FuncAnimation(
    fig, update,
    frames=frame_indices,
    interval=1000 // FPS,   # ms per frame
    blit=True
)

# ── Save animation ─────────────────────────────────────────────────────────
ani.save(os.path.join(os.path.dirname(__file__), "output/diffusion_animation.mp4"), writer='ffmpeg', fps=FPS)

# ── Create static plot of Squared Mean Distance over time ───────────────────────────────────────
smd = data['rRMS'] ** 2  # shape: (steps,)

plt.figure()
plt.plot(smd, label='Squared Mean Distance from Origin')
plt.xlabel('Step')
plt.ylabel('Squared Mean Distance (m²)')
plt.title('Squared Mean Distance from Origin Over Time')
plt.legend()
plt.grid()
plt.savefig(os.path.join(os.path.dirname(__file__), "output/Mean_Squared_Distance_plot.png"))

# __END__