"""
Celestial Dynamics Plotting and Animation Script
=================================================
Parses CSV output from celestial simulations and creates
2D/3D plots and animations of orbital paths.

Author: Generated for CPP_Workspace Project 3
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =============================================================================
# CONFIGURATION PARAMETERS - Easy to edit!
# =============================================================================

# File paths
OUTPUT_DIR = "/Users/nelsbuhrley/CPP_Workspace/Project 3: Celestial Dynamics/Output"
CSV_PATTERN = "celestial_output_{}.csv"  # {} will be replaced by index

# Plot display limits (in AU) - dynamic limits based on data
MIN_PLOT_RANGE = 1.0      # Minimum ±1 AU shown (even for small simulations)
MAX_PLOT_RANGE = 15.0     # Maximum ±15 AU shown (clips larger simulations)
PLOT_MARGIN = 0.05        # 5% margin beyond farthest point
Z_RANGE_FRACTION = 0.1    # Z-axis is this fraction of XY range (flatter view)

# Animation settings
ANIMATION_FPS = 20              # Frames per second (higher = smoother but larger file)
TARGET_FRAMES = 1000             # Target number of frames in animation (auto-calculates skip)
TRAIL_FRACTION = 0.018            # Fraction of animation to show as trail (0 = no trail, 1 = full)
ANIMATION_DPI = 100             # Lower DPI for faster rendering (100 is good balance)

# Plot appearance
FIGURE_SIZE_2D = (10, 10)
FIGURE_SIZE_3D = (12, 10)
FIGURE_SIZE_ANIM_2D = (8, 8)    # Smaller for faster animation rendering
FIGURE_SIZE_ANIM_3D = (10, 8)   # Smaller for faster animation rendering
MARKER_SIZE = 80            # Size of current position marker
TRAIL_LINEWIDTH = 1.5
TRAIL_ALPHA = 0.7

# Color scheme for celestial objects (add more as needed)
OBJECT_COLORS = {
    'Sun': '#FFD700',       # Gold
    'Mercury': '#A0522D',   # Sienna
    'Venus': '#DEB887',     # Burlywood
    'Earth': '#4169E1',     # Royal Blue
    'Mars': '#CD5C5C',      # Indian Red
    'Jupiter': '#DAA520',   # Goldenrod
    'Saturn': '#F4A460',    # Sandy Brown
    'Uranus': '#40E0D0',    # Turquoise
    'Neptune': '#1E90FF',   # Dodger Blue
    'Pluto': '#D3D3D3',     # Light Gray
    'Moon': '#C0C0C0',      # Silver
}

# Default color for unknown objects (will generate random if needed)
DEFAULT_COLOR = '#808080'

# Track dynamically assigned colors for unknown objects
_dynamic_colors = {}

# =============================================================================
# CSV PARSING FUNCTIONS
# =============================================================================

def parse_csv(filepath):
    """
    Parse a celestial dynamics CSV file.

    Returns:
        dict: {
            'objects': ['Sun', 'Mercury', ...],  # List of object names
            'time': np.array([...]),             # Time values
            'positions': {
                'Sun': {'x': [...], 'y': [...], 'z': [...]},
                'Mercury': {'x': [...], 'y': [...], 'z': [...]},
                ...
            }
        }
    """
    # Reset dynamic colors for each new file
    global _dynamic_colors
    _dynamic_colors = {}
    objects = []
    time_data = []
    positions = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First pass: extract object names from header comments
    for line in lines:
        if line.startswith('#Object:'):
            obj_name = line.split(':')[1].strip()
            objects.append(obj_name)
            positions[obj_name] = {'x': [], 'y': [], 'z': []}

    # If no objects found in comments, extract from column headers
    if not objects:
        for line in lines:
            if not line.startswith('#') and 'Time' in line:
                # Parse column header: Time,Sun_x,Sun_y,Sun_z,...
                cols = line.strip().split(',')
                for col in cols[1:]:  # Skip 'Time'
                    match = re.match(r'(.+)_[xyz]$', col)
                    if match:
                        obj_name = match.group(1)
                        if obj_name not in objects:
                            objects.append(obj_name)
                            positions[obj_name] = {'x': [], 'y': [], 'z': []}
                break

    # Second pass: read data
    for line in lines:
        if line.startswith('#') or 'Time' in line:
            continue

        values = line.strip().split(',')
        if len(values) < 1 + 3 * len(objects):
            continue

        try:
            time_data.append(float(values[0]))

            for i, obj in enumerate(objects):
                idx = 1 + i * 3  # Starting index for this object's x,y,z
                positions[obj]['x'].append(float(values[idx]))
                positions[obj]['y'].append(float(values[idx + 1]))
                positions[obj]['z'].append(float(values[idx + 2]))
        except (ValueError, IndexError):
            continue

    # Convert to numpy arrays
    for obj in objects:
        positions[obj]['x'] = np.array(positions[obj]['x'])
        positions[obj]['y'] = np.array(positions[obj]['y'])
        positions[obj]['z'] = np.array(positions[obj]['z'])

    return {
        'objects': objects,
        'time': np.array(time_data),
        'positions': positions
    }


def get_object_color(obj_name):
    """
    Get color for an object.
    Returns predefined color if available, otherwise generates a random
    unique color that hasn't been used yet.
    """
    import random

    # Check predefined colors first
    if obj_name in OBJECT_COLORS:
        return OBJECT_COLORS[obj_name]

    # Check if we already assigned a dynamic color to this object
    if obj_name in _dynamic_colors:
        return _dynamic_colors[obj_name]

    # Generate a random color that's not already in use
    used_colors = set(OBJECT_COLORS.values()) | set(_dynamic_colors.values())

    # Try to generate a distinct color
    for _ in range(100):  # Max attempts
        # Generate random hue, keep saturation and value high for visibility
        h = random.random()
        s = 0.6 + random.random() * 0.4  # 0.6-1.0 saturation
        v = 0.7 + random.random() * 0.3  # 0.7-1.0 value

        # Convert HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

        if color not in used_colors:
            _dynamic_colors[obj_name] = color
            return color

    # Fallback if we somehow can't find a unique color
    _dynamic_colors[obj_name] = DEFAULT_COLOR
    return DEFAULT_COLOR


def calculate_plot_limits(data):
    """
    Calculate dynamic plot limits based on simulation data.
    - Centered on origin
    - 5% margin beyond farthest point
    - Clamped between MIN_PLOT_RANGE and MAX_PLOT_RANGE

    Returns:
        tuple: (x_limits, y_limits, z_limits) each as (min, max)
    """
    # Find the maximum distance from origin across all objects
    max_distance = 0.0
    max_z = 0.0

    for obj in data['objects']:
        x = data['positions'][obj]['x']
        y = data['positions'][obj]['y']
        z = data['positions'][obj]['z']

        # Max distance in XY plane from origin
        distances = np.sqrt(x**2 + y**2)
        max_distance = max(max_distance, np.max(distances))

        # Max Z extent
        max_z = max(max_z, np.max(np.abs(z)))

    # Add margin and clamp
    xy_range = max_distance * (1 + PLOT_MARGIN)
    xy_range = max(MIN_PLOT_RANGE, min(MAX_PLOT_RANGE, xy_range))

    # Z range: proportional to XY but with its own minimum
    z_range = max(0.1, max_z * (1 + PLOT_MARGIN), xy_range * Z_RANGE_FRACTION)
    z_range = min(z_range, MAX_PLOT_RANGE * Z_RANGE_FRACTION)

    return ((-xy_range, xy_range),
            (-xy_range, xy_range),
            (-z_range, z_range))


# =============================================================================
# 2D PLOTTING FUNCTIONS
# =============================================================================

def plot_orbits_2d(data, title="Celestial Orbits (XY Plane)", save_path=None):
    """
    Create a static 2D plot of orbital paths in the XY plane.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_2D)

    # Calculate dynamic limits based on data
    x_limits, y_limits, _ = calculate_plot_limits(data)

    for obj in data['objects']:
        color = get_object_color(obj)
        x = data['positions'][obj]['x']
        y = data['positions'][obj]['y']

        # Plot orbital path
        ax.plot(x, y, color=color, linewidth=TRAIL_LINEWIDTH,
                alpha=TRAIL_ALPHA, label=obj)

        # Mark start and end positions
        ax.scatter(x[0], y[0], color=color, s=MARKER_SIZE,
                   marker='o', edgecolors='black', linewidth=1.5, zorder=5)
        ax.scatter(x[-1], y[-1], color=color, s=MARKER_SIZE/2,
                   marker='s', edgecolors='black', linewidth=1, alpha=0.7, zorder=5)

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel('X Position (AU)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (AU)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved 2D plot: {save_path}")

    plt.close()
    return fig


def animate_orbits_2d(data, title="Celestial Orbits Animation", save_path=None):
    """
    Create an animated 2D plot of orbital motion.
    Optimized for large datasets with automatic frame sampling.
    Uses FFmpeg for fast MP4 encoding.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_ANIM_2D)

    # Calculate dynamic limits based on data
    x_limits, y_limits, _ = calculate_plot_limits(data)

    # Auto-calculate frame skip based on data size and target frames
    n_points = len(data['time'])
    frame_skip = max(1, n_points // TARGET_FRAMES)
    indices = np.arange(0, n_points, frame_skip)
    n_frames = len(indices)

    # Calculate trail length in data points
    trail_points = int(n_points * TRAIL_FRACTION)

    # Pre-sample all position data for efficiency (avoid repeated slicing)
    sampled_data = {}
    for obj in data['objects']:
        sampled_data[obj] = {
            'x': data['positions'][obj]['x'][::frame_skip],
            'y': data['positions'][obj]['y'][::frame_skip]
        }

    # Initialize plot elements for each object
    trails = {}
    markers = {}

    for obj in data['objects']:
        color = get_object_color(obj)
        trails[obj], = ax.plot([], [], color=color, linewidth=TRAIL_LINEWIDTH,
                                alpha=TRAIL_ALPHA, label=obj)
        markers[obj] = ax.scatter([], [], color=color, s=MARKER_SIZE,
                                   edgecolors='black', linewidth=1.5, zorder=5)

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel('X Position (AU)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (AU)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)

    # Time display
    sampled_time = data['time'][::frame_skip]
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=11, verticalalignment='top',
                        fontfamily='monospace')

    # Trail length in sampled frames
    trail_frames = max(1, trail_points // frame_skip) if TRAIL_FRACTION > 0 else 0

    def init():
        for obj in data['objects']:
            trails[obj].set_data([], [])
            markers[obj].set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return list(trails.values()) + list(markers.values()) + [time_text]

    def update(frame):
        # Determine trail start
        start = max(0, frame - trail_frames) if trail_frames > 0 else 0

        for obj in data['objects']:
            x = sampled_data[obj]['x']
            y = sampled_data[obj]['y']

            # Update trail
            trails[obj].set_data(x[start:frame+1], y[start:frame+1])

            # Update current position marker
            markers[obj].set_offsets([[x[frame], y[frame]]])

        # Update time display
        time_text.set_text(f'Time: {sampled_time[frame]:.3f} years')

        return list(trails.values()) + list(markers.values()) + [time_text]

    interval_ms = 1000 // ANIMATION_FPS
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                         interval=interval_ms, blit=True)

    if save_path:
        # Change extension to .mp4 for FFmpeg
        save_path = save_path.replace('.gif', '.mp4')
        print(f"  Saving 2D animation ({n_frames} frames from {n_points} points)...")
        writer = FFMpegWriter(fps=ANIMATION_FPS, bitrate=1800)
        anim.save(save_path, writer=writer, dpi=ANIMATION_DPI)
        print(f"  Saved: {save_path}")

    plt.close()
    return anim


# =============================================================================
# 3D PLOTTING FUNCTIONS
# =============================================================================

def plot_orbits_3d(data, title="Celestial Orbits (3D View)", save_path=None):
    """
    Create a static 3D plot of orbital paths.
    """
    fig = plt.figure(figsize=FIGURE_SIZE_3D)
    ax = fig.add_subplot(111, projection='3d')

    # Calculate dynamic limits based on data
    x_limits, y_limits, z_limits = calculate_plot_limits(data)

    for obj in data['objects']:
        color = get_object_color(obj)
        x = data['positions'][obj]['x']
        y = data['positions'][obj]['y']
        z = data['positions'][obj]['z']

        # Plot orbital path
        ax.plot(x, y, z, color=color, linewidth=TRAIL_LINEWIDTH,
                alpha=TRAIL_ALPHA, label=obj)

        # Mark start and end positions
        ax.scatter(x[0], y[0], z[0], color=color, s=MARKER_SIZE,
                   marker='o', edgecolors='black', linewidth=1.5)
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=MARKER_SIZE/2,
                   marker='s', edgecolors='black', linewidth=1, alpha=0.7)

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)
    ax.set_xlabel('X Position (AU)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y Position (AU)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z Position (AU)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved 3D plot: {save_path}")

    plt.close()
    return fig


def animate_orbits_3d(data, title="Celestial Orbits Animation (3D)", save_path=None):
    """
    Create an animated 3D plot of orbital motion.
    Optimized for large datasets with automatic frame sampling.
    Uses FFmpeg for fast MP4 encoding.
    """
    fig = plt.figure(figsize=FIGURE_SIZE_ANIM_3D)
    ax = fig.add_subplot(111, projection='3d')

    # Calculate dynamic limits based on data
    x_limits, y_limits, z_limits = calculate_plot_limits(data)

    # Auto-calculate frame skip based on data size and target frames
    n_points = len(data['time'])
    frame_skip = max(1, n_points // TARGET_FRAMES)
    n_frames = n_points // frame_skip

    # Calculate trail length in data points
    trail_points = int(n_points * TRAIL_FRACTION)

    # Pre-sample all position data for efficiency
    sampled_data = {}
    for obj in data['objects']:
        sampled_data[obj] = {
            'x': data['positions'][obj]['x'][::frame_skip],
            'y': data['positions'][obj]['y'][::frame_skip],
            'z': data['positions'][obj]['z'][::frame_skip]
        }

    # Initialize plot elements
    trails = {}
    markers = {}

    for obj in data['objects']:
        color = get_object_color(obj)
        trails[obj], = ax.plot([], [], [], color=color, linewidth=TRAIL_LINEWIDTH,
                                alpha=TRAIL_ALPHA, label=obj)
        markers[obj] = ax.scatter([], [], [], color=color, s=MARKER_SIZE,
                                   edgecolors='black', linewidth=1.5)

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)
    ax.set_xlabel('X Position (AU)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y Position (AU)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z Position (AU)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Trail length in sampled frames
    trail_frames = max(1, trail_points // frame_skip) if TRAIL_FRACTION > 0 else 0

    def init():
        for obj in data['objects']:
            trails[obj].set_data_3d([], [], [])
            markers[obj]._offsets3d = ([], [], [])
        return list(trails.values()) + list(markers.values())

    def update(frame):
        # Determine trail start
        start = max(0, frame - trail_frames) if trail_frames > 0 else 0

        for obj in data['objects']:
            x = sampled_data[obj]['x']
            y = sampled_data[obj]['y']
            z = sampled_data[obj]['z']

            # Update trail
            trails[obj].set_data_3d(x[start:frame+1], y[start:frame+1], z[start:frame+1])

            # Update current position marker
            markers[obj]._offsets3d = ([x[frame]], [y[frame]], [z[frame]])

        return list(trails.values()) + list(markers.values())

    interval_ms = 1000 // ANIMATION_FPS
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                         interval=interval_ms, blit=False)

    if save_path:
        # Change extension to .mp4 for FFmpeg
        save_path = save_path.replace('.gif', '.mp4')
        print(f"  Saving 3D animation ({n_frames} frames from {n_points} points)...")
        writer = FFMpegWriter(fps=ANIMATION_FPS, bitrate=1800)
        anim.save(save_path, writer=writer, dpi=ANIMATION_DPI)
        print(f"  Saved: {save_path}")

    plt.close()
    return anim


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def get_csv_filepath(index):
    """Get the full filepath for a given CSV index."""
    filename = CSV_PATTERN.format(index)
    return os.path.join(OUTPUT_DIR, filename)


def process_csv(index, create_plots=True, create_animations=True):
    """
    Process a single CSV file and generate plots/animations.

    Args:
        index: The CSV file index number
        create_plots: Whether to create static plots
        create_animations: Whether to create animations

    Returns:
        dict: The parsed data, or None if file doesn't exist
    """
    filepath = get_csv_filepath(index)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    print(f"\nProcessing: {filepath}")

    # Parse the CSV
    data = parse_csv(filepath)
    print(f"  Found {len(data['objects'])} objects: {', '.join(data['objects'])}")
    print(f"  Time range: {data['time'][0]:.3f} to {data['time'][-1]:.3f} years")
    print(f"  Data points: {len(data['time'])}")

    base_name = f"celestial_analysis_{index}"

    if create_plots:
        # Static 2D plot
        plot_2d_path = os.path.join(OUTPUT_DIR, f"{base_name}_2d.png")
        plot_orbits_2d(data, title=f"Celestial Orbits - Simulation {index}",
                       save_path=plot_2d_path)

        # Static 3D plot
        plot_3d_path = os.path.join(OUTPUT_DIR, f"{base_name}_3d.png")
        plot_orbits_3d(data, title=f"Celestial Orbits (3D) - Simulation {index}",
                       save_path=plot_3d_path)

    if create_animations:
        # 2D animation
        anim_2d_path = os.path.join(OUTPUT_DIR, f"{base_name}_2d.gif")
        animate_orbits_2d(data, title=f"Celestial Motion - Simulation {index}",
                          save_path=anim_2d_path)

        # 3D animation
        anim_3d_path = os.path.join(OUTPUT_DIR, f"{base_name}_3d.gif")
        animate_orbits_3d(data, title=f"Celestial Motion (3D) - Simulation {index}",
                          save_path=anim_3d_path)

    return data


def has_all_outputs(index):
    """
    Check if all output files (plots + animations) exist for a given CSV index.
    Returns True if ALL outputs exist, False if any are missing.
    """
    base_name = f"celestial_analysis_{index}"
    required_files = [
        f"{base_name}_2d.png",
        f"{base_name}_3d.png",
        f"{base_name}_2d.mp4",
        f"{base_name}_3d.mp4"
    ]

    for filename in required_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            return False
    return True


def find_next_unprocessed(max_files=100):
    """
    Find the lowest index CSV file that doesn't have all its outputs generated.
    Returns the index, or None if all CSVs are processed or no CSVs exist.
    """
    for i in range(max_files):
        csv_path = get_csv_filepath(i)
        if os.path.exists(csv_path):
            if not has_all_outputs(i):
                return i
        else:
            # No more CSV files in sequence
            break
    return None


def process_next_csv(create_plots=True, create_animations=True):
    """
    Process only the next unprocessed CSV file (lowest index without all outputs).
    This is the main entry point for incremental processing.

    Returns:
        dict: The parsed data, or None if no unprocessed CSVs found
    """
    print("=" * 60)
    print("Celestial Dynamics Visualization")
    print("=" * 60)

    index = find_next_unprocessed()

    if index is None:
        print("\nAll CSV files have been processed!")
        print("(No unprocessed celestial_output_*.csv files found)")
        return None

    print(f"\nFound unprocessed CSV: index {index}")
    result = process_csv(index, create_plots, create_animations)

    print(f"\n{'=' * 60}")
    print(f"Completed processing celestial_output_{index}.csv")
    print("=" * 60)

    return result


def process_all_csvs(create_plots=True, create_animations=True, max_files=100):
    """
    Process all available CSV files in the output directory.
    Skips files that already have all outputs generated.

    Args:
        create_plots: Whether to create static plots
        create_animations: Whether to create animations
        max_files: Maximum number of files to process
    """
    print("=" * 60)
    print("Celestial Dynamics Visualization (Process All)")
    print("=" * 60)

    processed = 0
    skipped = 0
    for i in range(max_files):
        filepath = get_csv_filepath(i)
        if os.path.exists(filepath):
            if has_all_outputs(i):
                print(f"  Skipping index {i} (already processed)")
                skipped += 1
            else:
                process_csv(i, create_plots, create_animations)
                processed += 1
        else:
            # Stop when we hit a gap in the sequence
            if i > 0:
                break

    print(f"\n{'=' * 60}")
    print(f"Processed {processed} CSV files, skipped {skipped}")
    print("=" * 60)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Process all CSVs (skip already processed)
        if arg == '--all':
            process_all_csvs(create_plots=True, create_animations=True)

        # If a specific file is provided, process just that one
        elif arg.endswith('.csv'):
            filepath = arg
            if os.path.exists(filepath):
                data = parse_csv(filepath)
                # Extract index from filename if possible
                match = re.search(r'celestial_output_?(\d+)', filepath)
                idx = int(match.group(1)) if match else 0

                base_name = f"celestial_analysis_{idx}"
                output_dir = os.path.dirname(filepath)

                plot_orbits_2d(data, save_path=os.path.join(output_dir, f"{base_name}_2d.png"))
                plot_orbits_3d(data, save_path=os.path.join(output_dir, f"{base_name}_3d.png"))
                animate_orbits_2d(data, save_path=os.path.join(output_dir, f"{base_name}_2d.gif"))
                animate_orbits_3d(data, save_path=os.path.join(output_dir, f"{base_name}_3d.gif"))
            else:
                print(f"File not found: {filepath}")

        # Process specific index
        else:
            try:
                idx = int(arg)
                process_csv(idx, create_plots=True, create_animations=True)
            except ValueError:
                print("Usage:")
                print("  python plotting.py           # Process next unprocessed CSV")
                print("  python plotting.py 3         # Process specific index")
                print("  python plotting.py --all     # Process all unprocessed CSVs")
                print("  python plotting.py file.csv  # Process specific file")
    else:
        # Default: process only the NEXT unprocessed CSV
        process_next_csv(create_plots=True, create_animations=True)
