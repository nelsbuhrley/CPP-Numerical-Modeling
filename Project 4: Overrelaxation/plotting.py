import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

data = np.load("output.npz")
potential = data["potential"]
center = potential.shape[0] // 2

# Create figure and 3D axis
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for X and Y positions (physical coordinates in meters)
X = np.linspace(-0.5, 0.5, potential.shape[0])
Y = np.linspace(-0.5, 0.5, potential.shape[1])
X, Y = np.meshgrid(X, Y)

# Initial surface plot
Z = potential[center, :, :]
surf = ax.plot_surface(X, Y, Z, cmap='viridis', vmin=potential.min(), vmax=potential.max())
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Potential (V)')
ax.set_title(f'Potential Distribution at Z Slice {center}')

# Create colorbar ONCE
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Create slider
ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03])
z_slider = Slider(ax_slider, 'Z Slice', 0, potential.shape[0] - 1, valinit=center, valstep=1)

def update(val):
    z_index = int(z_slider.val)

    # Remove all existing surfaces
    while ax.collections:
        ax.collections[0].remove()

    # Load NEW slice data
    Z = potential[z_index, :, :]
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', vmin=potential.min(), vmax=potential.max())

    ax.set_title(f'Potential Distribution at Z Slice {z_index}')

    fig.canvas.draw_idle()

z_slider.on_changed(update)
plt.show()