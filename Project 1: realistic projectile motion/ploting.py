import matplotlib.pyplot as plt
import pandas as pd
import sys


# Should automatically share run when running from cpp workspace
# open csv file in with the path
# f/Users/nelsbuhrley/CPP_Workspace/Project 1: realistic projectile motion/Output/trajectory{i}.csv
# ask the user for "i" value
i = input("Enter the trajectory index (i): ")
file_path = f"/Users/nelsbuhrley/CPP_Workspace/Project 1: realistic projectile motion/Output/trajectory{i}.csv"

def plot_trajectory(file_path):
    """Read and plot projectile trajectory data from CSV file.
        This is a 3D projectile motion plotter. the csv file has columns: time, x, y, z
    """
    # Read CSV file
    # Skip comment lines starting with '#'
    data = pd.read_csv(file_path, comment='#')
    print(1)
    # Create 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(data['X'], data['Y'], data['Z'], label='Projectile Trajectory', color='b')

    # and the first and last points
    ax.scatter(data['X'].iloc[0], data['Y'].iloc[0], data['Z'].iloc[0], color='g', s=50, label='Start Point')
    ax.scatter(data['X'].iloc[-1], data['Y'].iloc[-1], data['Z'].iloc[-1], color='r', s=50, label='End Point')

    # Labels and title
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    ax.set_title('3D Projectile Motion Trajectory', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3)

    # Show legend
    ax.legend()

    # Show plot
    plt.show()

    # save plot as png in the same directory as the csv file
    output_path = file_path.replace('.csv', '.png')
    fig.savefig(output_path)
    print(f"Plot saved as '{output_path}'")


if __name__ == "__main__":
    plot_trajectory(file_path)