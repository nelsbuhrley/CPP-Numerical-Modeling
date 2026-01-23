#!/usr/bin/env python3
"""
Advanced Oscillator plotting Tool
Designed to be called by other programs or run interactively.
Creates 4 plots: angle vs time, wrapped angle vs time, phase space, and Poincaré section.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re


def wrap_angle(angle):
    """Wrap angle to [-π, π] range"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

def extract_number_from_filename(filepath):
    """Extract the number from the filename (e.g., oscillator_output_4.csv -> 4)"""
    match = re.search(r'oscillator_output_?(\d+)', filepath)
    if match:
        return match.group(1)
    return "output"

def get_parameters():
    """Get parameters from command line arguments or user input"""
    if len(sys.argv) >= 4:
        # Parameters provided via command line
        csv_filepath = sys.argv[1]
        max_plot_time = float(sys.argv[2])
        driver_frequency = float(sys.argv[3])
    else:
        # Interactive mode - prompt user for input
        print("=" * 60)
        print("Advanced Oscillator plotting Tool")
        print("=" * 60)
        print()

        csv_filepath = input("Enter CSV file path (e.g., /path/to/Output/oscillator_output_1.csv): ").strip()
        max_plot_time = float(input("Enter maximum plot time for first 3 plots (seconds): ").strip())
        driver_frequency = float(input("Enter driver frequency (rad/s): ").strip())

    return csv_filepath, max_plot_time, driver_frequency

def load_data(csv_filepath):
    """Load and validate CSV data"""
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")

    data = pd.read_csv(csv_filepath, comment='#')

    # Validate required columns
    required_cols = ['Time', 'Angle', 'AngularVelocity']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    return data

def create_plots(data, max_plot_time, driver_frequency, output_filepath):
    """Create all 4 plots in a single figure"""

    # Filter data for first 180 seconds (or specified time)
    data_filtered = data[data['Time'] <= max_plot_time].copy()

    # Wrap angles for filtered data
    data_filtered['AngleWrapped'] = wrap_angle(data_filtered['Angle'])

    # Calculate Poincaré section sampling droping the last data point if it exceeds max time
    driver_period = 2 * np.pi / driver_frequency
    poincare_times = np.arange(0, data['Time'].max(), driver_period)[:-1]


    # Find closest data points to Poincaré sampling times
    poincare_indices = np.searchsorted(data['Time'].values, poincare_times)
    poincare_indices_ofset_pi_over_2 = np.searchsorted(data['Time'].values, poincare_times + driver_period / 4)
    poincare_indices_ofset_pi_over_4 = np.searchsorted(data['Time'].values, poincare_times + driver_period / 8)

    poincare_data = data.iloc[poincare_indices].copy()
    poincare_data['AngleWrapped'] = wrap_angle(poincare_data['Angle'])

    poincare_data_ofset_pi_over_2 = data.iloc[poincare_indices_ofset_pi_over_2].copy()
    poincare_data_ofset_pi_over_2['AngleWrapped'] = wrap_angle(poincare_data_ofset_pi_over_2['Angle'])
    poincare_data_ofset_pi_over_4 = data.iloc[poincare_indices_ofset_pi_over_4].copy()
    poincare_data_ofset_pi_over_4['AngleWrapped'] = wrap_angle(poincare_data_ofset_pi_over_4['Angle'])

    # Create figure with 6 subplots
    fig = plt.figure(figsize=(18, 15))

    # Plot 1: Angle vs Time (0-180s)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(data_filtered['Time'], data_filtered['Angle'], linewidth=1, color='#1f77b4')
    ax1.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Angle (rad)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Angle vs Time (0-{max_plot_time}s)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max_plot_time)

    # Plot 2: Wrapped Angle vs Time (0-180s)
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(data_filtered['Time'], data_filtered['AngleWrapped'], linewidth=1, color='#ff7f0e')

    # Add horizontal reference lines at -π, 0, π
    for n in [-1, 0, 1]:
        ax2.axhline(y=n*np.pi, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Angle (rad)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Wrapped Angle vs Time (0-{max_plot_time}s)', fontsize=13, fontweight='bold')
    ax2.set_ylim(-np.pi * 1.1, np.pi * 1.1)
    ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, max_plot_time)

    # Plot 3: Phase Space (0-180s)
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(data_filtered['AngleWrapped'], data_filtered['AngularVelocity'],
             linewidth=1, color='#2ca02c', alpha=0.7)
    ax3.set_xlabel('Angle (rad)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Phase Space (0-{max_plot_time}s)', fontsize=13, fontweight='bold')
    ax3.set_xlim(-np.pi * 1.05, np.pi * 1.05)
    ax3.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax3.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Add vertical reference lines
    for n in [-1, 0, 1]:
        ax3.axvline(x=n*np.pi, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Plot 4: Poincaré Section (full dataset)
    ax4 = plt.subplot(3, 2, 4)
    ax4.scatter(poincare_data['AngleWrapped'], poincare_data['AngularVelocity'],
                s=15, color='#d62728', alpha=0.6, edgecolors='black', linewidth=0.3)
    ax4.set_xlabel('Angle (rad)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Poincaré Section (T = {driver_period:.3f}s, Full Dataset)',
                  fontsize=13, fontweight='bold')
    ax4.set_xlim(-np.pi * 1.05, np.pi * 1.05)
    ax4.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax4.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax4.grid(True, alpha=0.3, linestyle='--')

    #plot 5: Poincaré Section Offset π/2 (full dataset)
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.scatter(poincare_data_ofset_pi_over_2['AngleWrapped'], poincare_data_ofset_pi_over_2['AngularVelocity'],
                s=15, color='#9467bd', alpha=0.6, edgecolors='black', linewidth=0.3)
    ax5.set_xlabel('Angle (rad)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
    ax5.set_title(f'Poincaré Section Offset π/2 (T = {driver_period:.3f}s, Full Dataset)',
                  fontsize=13, fontweight='bold')
    ax5.set_xlim(-np.pi * 1.05, np.pi * 1.05)
    ax5.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax5.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax5.grid(True, alpha=0.3, linestyle='--')

    # Plot 6: Poincaré Section Offset π/4 (full dataset)
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.scatter(poincare_data_ofset_pi_over_4['AngleWrapped'], poincare_data_ofset_pi_over_4['AngularVelocity'],
                s=15, color='#8c564b', alpha=0.6, edgecolors='black', linewidth=0.3)
    ax6.set_xlabel('Angle (rad)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
    ax6.set_title(f'Poincaré Section Offset π/4 (T = {driver_period:.3f}s, Full Dataset)',
                  fontsize=13, fontweight='bold')
    ax6.set_xlim(-np.pi * 1.05, np.pi * 1.05)
    ax6.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax6.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax6.grid(True, alpha=0.3, linestyle='--')

    # Add vertical reference lines
    for n in [-1, 0, 1]:
        ax4.axvline(x=n*np.pi, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax5.axvline(x=n*np.pi, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax6.axvline(x=n*np.pi, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Add main title
    fig.suptitle('Driven Damped Oscillator Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save the figure
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_filepath}")

    # Display the figure
    plt.show()

def main():
    """Main execution function"""
    try:
        # Get parameters
        csv_filepath, max_plot_time, driver_frequency = get_parameters()

        # Extract number from input filename
        file_number = extract_number_from_filename(csv_filepath)

        # Construct output filepath
        output_dir = os.path.join(os.path.dirname(csv_filepath))
        output_filepath = os.path.join(output_dir, f'oscillator_analysis_{file_number}.png')

        print(f"\nLoading data from: {csv_filepath}")
        print(f"Max plot time: {max_plot_time}s")
        print(f"Driver frequency: {driver_frequency} rad/s")
        print(f"Driver period: {2*np.pi/driver_frequency:.3f}s")

        # Load data
        data = load_data(csv_filepath)
        print(f"Loaded {len(data)} data points")
        print(f"Time range: {data['Time'].min():.2f}s to {data['Time'].max():.2f}s")

        # Create plots
        create_plots(data, max_plot_time, driver_frequency, output_filepath)

        print("\nplotting complete!")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
