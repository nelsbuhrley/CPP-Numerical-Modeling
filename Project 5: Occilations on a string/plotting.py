import numpy as np
import matplotlib.pyplot as plt

# Load the data (saved as npy format despite .npz extension)
data = np.load("string_oscillations.npz")

# Shape: (timeSteps, segments) = (1000, 100)
print(f"Data shape: {data.shape}")

# String parameters (must match main.cpp)
length = 1.0
segments = data.shape[1]
x = np.linspace(0, length, segments)

# Plot the first timestep (initial condition)

for t in range(0, 2000, 100):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, data[t], color="steelblue", linewidth=2, label=f"t = {t}")
    ax.set_xlabel("Position along string (m)", fontsize=13)
    ax.set_ylabel("Displacement", fontsize=13)
    ax.set_title(f"String Displacement at Timestep t = {t}", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, length)
    plt.tight_layout()
    plt.show()
print("Plot saved to string_t0.png")
