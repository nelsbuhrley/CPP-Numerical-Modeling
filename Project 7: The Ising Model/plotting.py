import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# ===== 3D Plot Viewing Angle Parameters =====
# Elevation angle (default: 25 degrees)
VIEW_ELEVATION = 25
# Azimuth angle (default: 45 degrees) - rotate around vertical axis
VIEW_AZIMUTH = 45

# Set to True to save multiple rotations of the 3D plot
SAVE_MULTIPLE_ANGLES = False

# Angles to save if SAVE_MULTIPLE_ANGLES is True
ROTATION_ANGLES = [(25, 45), (25, 135), (25, 225), (25, 315)]

# Load the npz file
data = np.load('output/ising_results.npz')

temperatures = data['temperatures']
magnetic_fields = data['magnetic_fields']
magnetizations = data['avg_magnetizations']

# Reshape magnetization data to 2D
# Data is saved as (h_values.size, temps.size)
magnetizations_2d = magnetizations.reshape(len(magnetic_fields), len(temperatures))

# Create meshgrid for 3D plotting
T_mesh, H_mesh = np.meshgrid(temperatures, magnetic_fields)

# Function to create and save 3D plot with specific viewing angle
def save_3d_plot(elev, azim, output_suffix=''):
    """Save 3D surface plot with specified viewing angle."""
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(T_mesh, H_mesh, magnetizations_2d, cmap='viridis', alpha=0.85, edgecolor='none')

    ax.set_xlabel('Temperature (T)', fontsize=12, labelpad=10)
    ax.set_ylabel('Magnetic Field (h)', fontsize=12, labelpad=10)
    ax.set_zlabel('Average Magnetization', fontsize=12, labelpad=10)
    ax.set_title('3D Ising Model: Magnetization vs Temperature and Magnetic Field', fontsize=14, pad=20)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, label='Magnetization', shrink=0.6, pad=0.1)

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    plt.show()
    # Save the 3D plot
    filename = f'output/magnetization_3d_surface{output_suffix}.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f"3D surface plot saved to {filename} (elev={elev}°, azim={azim}°)")
    plt.close()

# Save 3D plot(s) based on configuration
if SAVE_MULTIPLE_ANGLES:
    print(f"\nSaving 3D plots from {len(ROTATION_ANGLES)} different angles:")
    for i, (elev, azim) in enumerate(ROTATION_ANGLES):
        save_3d_plot(elev, azim, f'_angle{i+1}')
else:
    save_3d_plot(VIEW_ELEVATION, VIEW_AZIMUTH)


# Also create a 2D contour plot
fig2, ax2 = plt.subplots(figsize=(11, 8))
levels = np.linspace(magnetizations_2d.min(), magnetizations_2d.max(), 20)
contour = ax2.contourf(T_mesh, H_mesh, magnetizations_2d, levels=levels, cmap='viridis')
contour_lines = ax2.contour(T_mesh, H_mesh, magnetizations_2d, levels=levels, colors='black', alpha=0.2, linewidths=0.5)

ax2.set_xlabel('Temperature (T)', fontsize=12)
ax2.set_ylabel('Magnetic Field (h)', fontsize=12)
ax2.set_title('Ising Model: Magnetization Contour Map', fontsize=14)
cbar2 = fig2.colorbar(contour, ax=ax2, label='Magnetization')
ax2.clabel(contour_lines, inline=True, fontsize=8)

plt.tight_layout()

# Save the contour plot
plt.savefig('output/magnetization_contour.png', dpi=300, bbox_inches='tight')
print("Contour plot saved to output/magnetization_contour.png")

plt.show()
