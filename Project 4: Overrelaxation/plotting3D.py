import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Load Data
data = np.load("output.npz")
potential = data["potential"]
depth, rows, cols = potential.shape
center = depth // 2

# 2. Setup Figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# 3. Create Meshgrids - Full resolution for saving, downsampled for interactive
x_full = np.linspace(-0.5, 0.5, rows)
y_full = np.linspace(-0.5, 0.5, cols)
X_full, Y_full = np.meshgrid(x_full, y_full)

# Downsampled meshgrid for interactive slider (75x75)
x_low = np.linspace(-0.5, 0.5, 75)
y_low = np.linspace(-0.5, 0.5, 75)
X_low, Y_low = np.meshgrid(x_low, y_low)

# 4. Initial Plot with 75x75 resolution
from scipy.ndimage import zoom
zoom_factor = 75 / rows
Z_low = zoom(potential[center, :, :], zoom_factor, order=1)
surf = ax.plot_surface(X_low, Y_low, Z_low, cmap='viridis',
                       vmin=potential.min(), vmax=potential.max(),
                       antialiased=False)

# Lock the axes so they don't jump
ax.set_zlim(potential.min(), potential.max())
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Potential (V)')

# 5. Fast Update Function with downsampling
def update(val):
    z_index = int(z_slider.val)
    new_Z_full = potential[z_index, :, :]

    # Downsample to 75x75 for fast rendering
    new_Z_low = zoom(new_Z_full, zoom_factor, order=1)

    # High-speed update: We modify the internal vertex data directly
    # This avoids the overhead of removing and recreating the surface
    global surf
    surf.remove()
    surf = ax.plot_surface(X_low, Y_low, new_Z_low, cmap='viridis',
                           vmin=potential.min(), vmax=potential.max(),
                           antialiased=False)

    fig.canvas.draw_idle()

# 6. Slider Setup
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
z_slider = Slider(ax_slider, 'Z Slice', 0, depth - 1, valinit=center, valstep=1)
z_slider.on_changed(update)

# Save 3D plot at center slice before showing interactive plot (FULL RESOLUTION)
fig_save = plt.figure(figsize=(12, 9))
ax_save = fig_save.add_subplot(111, projection='3d')

Z_center = potential[center, :, :]
surf_save = ax_save.plot_surface(X_full, Y_full, Z_center, cmap='viridis',
                                  vmin=potential.min(), vmax=potential.max(),
                                  antialiased=True, alpha=0.9)

ax_save.set_xlabel('X Position (m)', fontsize=12)
ax_save.set_ylabel('Y Position (m)', fontsize=12)
ax_save.set_zlabel('Potential (V)', fontsize=12)
ax_save.set_title(f'3D Potential Field at Z = {center} (Center Slice)', fontsize=14)
ax_save.set_zlim(potential.min(), potential.max())

# Add colorbar
cbar = fig_save.colorbar(surf_save, ax=ax_save, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('Potential (V)', fontsize=12)

# Set a nice viewing angle
ax_save.view_init(elev=25, azim=45)

# Save high-quality image
plt.savefig('potential_3D_center_slice.png', dpi=300, bbox_inches='tight')
print(f"Saved 3D plot at z={center} to 'potential_3D_center_slice.png'")
plt.close(fig_save)

plt.show()
