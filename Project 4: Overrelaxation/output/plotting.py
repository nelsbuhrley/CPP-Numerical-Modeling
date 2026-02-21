import numpy as np
import matplotlib.pyplot as plt

"""
Header

This script loads the 3D potential field data from 'output.npz' and creates
 a high-resolution 3D surface plot of the center slice (z = N/2).
 The plot is interactive, allowing you to rotate it to your desired angle
 before saving. The final image is saved as 'potential_3D_center_slice.png'
 with a resolution of 300 DPI.

Author: Nels Buhrley
Date: 2024-06-01
"""

# Load data
data = np.load("output.npz")
potential = data["potential"]
center = potential.shape[0] // 2

# Create high-resolution figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for full resolution
X = np.linspace(-0.5, 0.5, potential.shape[0])
Y = np.linspace(-0.5, 0.5, potential.shape[1])
X, Y = np.meshgrid(X, Y)

# Plot center slice at full resolution
Z = potential[center, :, :]
surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                       vmin=potential.min(), vmax=potential.max(),
                       antialiased=True, alpha=0.9)

ax.set_xlabel('X Position (m)', fontsize=12)
ax.set_ylabel('Y Position (m)', fontsize=12)
ax.set_zlabel('Potential (V)', fontsize=12)
ax.set_title(f'3D Potential Field at Z = {center} (Center Slice)', fontsize=14)

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Potential (V)', fontsize=12)

# Set viewing angle
ax.view_init(elev=25, azim=45)

# Display (rotate to desired angle, then close window)
print("Rotate the plot to your desired angle, then close the window to save...")
plt.show()

# Save high-resolution image with the angle you chose
fig.savefig('potential_3D_center_slice.png', dpi=300, bbox_inches='tight')
print(f"Saved 3D plot at z={center} to 'potential_3D_center_slice.png'")