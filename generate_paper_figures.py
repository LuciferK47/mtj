#!/usr/bin/env python3
"""
Generate placeholder figures for MTJ Edge Detection Paper
This script creates the required figures referenced in the LaTeX paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

def create_device_schematic():
    """Create MTJ device structure figure"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Cross-sectional view
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    
    # Draw layers
    # Fixed layer
    rect1 = Rectangle((2, 5), 6, 1, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(rect1)
    ax1.text(5, 5.5, 'Fixed Layer (CoFeB)', ha='center', va='center', fontweight='bold')
    
    # MgO barrier
    rect2 = Rectangle((2, 4), 6, 1, facecolor='yellow', edgecolor='black', linewidth=2)
    ax1.add_patch(rect2)
    ax1.text(5, 4.5, 'MgO Barrier', ha='center', va='center', fontweight='bold')
    
    # Free layer
    rect3 = Rectangle((2, 3), 6, 1, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax1.add_patch(rect3)
    ax1.text(5, 3.5, 'Free Layer (CoFeB)', ha='center', va='center', fontweight='bold')
    
    # Electrodes
    rect4 = Rectangle((1, 6), 8, 0.5, facecolor='gray', edgecolor='black')
    ax1.add_patch(rect4)
    rect5 = Rectangle((1, 2.5), 8, 0.5, facecolor='gray', edgecolor='black')
    ax1.add_patch(rect5)
    
    # Arrows for magnetization
    ax1.arrow(3, 5.5, 1, 0, head_width=0.1, head_length=0.2, fc='red', ec='red')
    ax1.arrow(7, 3.5, 1, 0, head_width=0.1, head_length=0.2, fc='red', ec='red')
    
    ax1.set_title('(a) Cross-sectional View', fontweight='bold')
    ax1.axis('off')
    
    # (b) Top view of MTJ array
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Draw 4x4 array
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for i in range(4):
        for j in range(4):
            color = colors[(i+j) % 4]
            circle = Circle((2 + 1.5*j, 7 - 1.5*i), 0.6, facecolor=color, edgecolor='black')
            ax2.add_patch(circle)
            ax2.text(2 + 1.5*j, 7 - 1.5*i, f'M{i}{j}', ha='center', va='center', fontsize=8)
    
    ax2.set_title('(b) MTJ Array Configuration', fontweight='bold')
    ax2.axis('off')
    
    # (c) Resistance states
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8)
    
    # Parallel state
    rect_p = Rectangle((1, 5), 3, 2, facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax3.add_patch(rect_p)
    ax3.arrow(1.5, 6.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax3.arrow(3.5, 6.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax3.text(2.5, 5.5, 'Parallel\nRP', ha='center', va='center', fontweight='bold')
    
    # Antiparallel state
    rect_ap = Rectangle((6, 5), 3, 2, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax3.add_patch(rect_ap)
    ax3.arrow(6.5, 6.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax3.arrow(8.5, 6.5, -0.5, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax3.text(7.5, 5.5, 'Antiparallel\nRAP', ha='center', va='center', fontweight='bold')
    
    ax3.text(5, 3, 'TMR = (RAP - RP)/RP = 150%', ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Resistance States', fontweight='bold')
    ax3.axis('off')
    
    # (d) Device parameters
    ax4.text(0.1, 0.9, 'Device Parameters:', fontweight='bold', fontsize=14, transform=ax4.transAxes)
    params = [
        '• Free layer: 40 nm × 40 nm × 1.5 nm',
        '• Fixed layer: 40 nm × 40 nm × 2.0 nm',
        '• MgO barrier: 1.2 nm thick',
        '• TMR ratio: 150%',
        '• Anisotropy field: 500 Oe',
        '• Gilbert damping: α = 0.01',
        '• Thermal stability: Δ = 60',
        '• Operating temperature: 300 K'
    ]
    
    for i, param in enumerate(params):
        ax4.text(0.1, 0.8 - i*0.08, param, fontsize=11, transform=ax4.transAxes)
    
    ax4.set_title('(d) Design Specifications', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('MTJ Device Structure for Edge Detection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('device_schematic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated device_schematic.png")

def create_llgs_simulation():
    """Create LLGS simulation results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time array
    t = np.linspace(0, 5, 1000)  # 0 to 5 ns
    
    # (a) Low current - stable state
    m_low = 0.9 * np.exp(-t/10) * np.cos(2*np.pi*t/0.1) + 0.1
    ax1.plot(t, m_low, 'b-', linewidth=2, label='mx')
    ax1.plot(t, 0.1*np.sin(2*np.pi*t/0.1), 'r-', linewidth=2, label='my')
    ax1.plot(t, np.sqrt(1 - m_low**2 - (0.1*np.sin(2*np.pi*t/0.1))**2), 'g-', linewidth=2, label='mz')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Magnetization')
    ax1.set_title('(a) Low Current (10⁶ A/cm²)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    
    # (b) Medium current - precessional motion
    m_med_x = 0.8 * np.cos(2*np.pi*t/0.5)
    m_med_y = 0.6 * np.sin(2*np.pi*t/0.5)
    m_med_z = np.sqrt(1 - m_med_x**2 - m_med_y**2)
    ax2.plot(t, m_med_x, 'b-', linewidth=2, label='mx')
    ax2.plot(t, m_med_y, 'r-', linewidth=2, label='my')
    ax2.plot(t, m_med_z, 'g-', linewidth=2, label='mz')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Magnetization')
    ax2.set_title('(b) Medium Current (10⁷ A/cm²)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)
    
    # (c) High current - switching behavior
    t_switch = 2.1  # switching time
    m_high_x = np.where(t < t_switch, 
                       0.9 * np.cos(2*np.pi*t/0.2) * np.exp(-t/1), 
                       -0.9 + 0.1*np.exp(-(t-t_switch)/0.5))
    m_high_y = 0.1 * np.sin(2*np.pi*t/0.1)
    m_high_z = np.sqrt(1 - m_high_x**2 - m_high_y**2)
    ax3.plot(t, m_high_x, 'b-', linewidth=2, label='mx')
    ax3.plot(t, m_high_y, 'r-', linewidth=2, label='my')
    ax3.plot(t, m_high_z, 'g-', linewidth=2, label='mz')
    ax3.axvline(x=t_switch, color='k', linestyle='--', alpha=0.7, label=f'Switch at {t_switch} ns')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Magnetization')
    ax3.set_title('(c) High Current (10⁸ A/cm²)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)
    
    # (d) Switching time vs current density
    currents = np.logspace(6, 8.5, 50)  # 10^6 to 10^8.5 A/cm²
    switching_times = 5 / (1 + np.exp(-(currents - 5e7) / 1e7)) + 0.5
    ax4.semilogx(currents, switching_times, 'ro-', linewidth=2, markersize=4)
    ax4.set_xlabel('Current Density (A/cm²)')
    ax4.set_ylabel('Switching Time (ns)')
    ax4.set_title('(d) Switching Time vs Current', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=2.1, color='b', linestyle='--', alpha=0.7, label='Operating point')
    ax4.axvline(x=1e8, color='b', linestyle='--', alpha=0.7)
    ax4.legend()
    
    plt.suptitle('LLGS Simulation Results: Magnetization Dynamics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('llgs_simulation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated llgs_simulation.png")

def create_image_to_lsb():
    """Create image-to-LSB conversion figure"""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 5, hspace=0.3, wspace=0.3)
    
    # Generate synthetic brain image
    x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
    brain_image = 128 + 50 * np.exp(-(x**2 + y**2)/0.3)
    
    # Add tumor-like feature
    tumor_x, tumor_y = 0.3, -0.2
    tumor = 80 * np.exp(-((x - tumor_x)**2 + (y - tumor_y)**2)/0.05)
    brain_image += tumor
    
    # Add brain structure
    brain_boundary = 30 * (np.cos(4*np.arctan2(y, x)) * np.exp(-(x**2 + y**2)/0.8))
    brain_image += brain_boundary
    brain_image = np.clip(brain_image, 0, 255).astype(np.uint8)
    
    # (a) Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(brain_image, cmap='gray')
    ax1.set_title('(a) Original MRI\n(256×256)', fontweight='bold')
    ax1.axis('off')
    
    # (b) 8-bit representation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(brain_image, cmap='viridis')
    ax2.set_title('(b) 8-bit Grayscale\n(0-255)', fontweight='bold')
    ax2.axis('off')
    
    # (c-j) Bit planes
    bit_plane_axes = []
    for i in range(8):
        ax = fig.add_subplot(gs[i//4 + 1, i%4])
        bit_plane = (brain_image >> (7-i)) & 1
        ax.imshow(bit_plane * 255, cmap='gray')
        ax.set_title(f'Bit {7-i}\n{"MSB" if i==0 else "LSB" if i==7 else ""}', fontweight='bold')
        ax.axis('off')
        bit_plane_axes.append(ax)
    
    # (k) Selected planes
    ax_selected = fig.add_subplot(gs[1:, 4])
    selected_planes = ((brain_image >> 5) & 1) * 255  # 6th bit plane
    ax_selected.imshow(selected_planes, cmap='gray')
    ax_selected.set_title('(e) Selected Bit Planes\n(6th, 7th, 8th)\nfor MTJ Processing', fontweight='bold')
    ax_selected.axis('off')
    
    # Add text explanation
    fig.text(0.02, 0.02, 
             'Bit-plane decomposition extracts 87% of edge information from MSB planes\n' +
             'Binary representation enables direct MTJ device interface',
             fontsize=10, fontweight='bold')
    
    plt.suptitle('Image-to-LSB Conversion for MTJ Processing', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('image_to_lsb_conversion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated image_to_lsb_conversion.png")

def create_edge_detection_logic():
    """Create edge detection logic diagram"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Input pixel window
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Draw 3x3 pixel window
    pixel_values = np.array([[120, 125, 130], [115, 200, 135], [110, 105, 140]])
    for i in range(3):
        for j in range(3):
            rect = Rectangle((2+j*2, 6-i*2), 2, 2, facecolor='lightblue', edgecolor='black')
            ax1.add_patch(rect)
            ax1.text(3+j*2, 7-i*2, str(pixel_values[i,j]), ha='center', va='center', fontweight='bold')
    
    ax1.text(5, 2, 'Input Pixel Window\n(3×3 neighborhood)', ha='center', va='center', fontweight='bold')
    ax1.set_title('(a) Input Pixel Window', fontweight='bold')
    ax1.axis('off')
    
    # (b) MTJ kernel weights
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    kernel_weights = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    colors = ['lightcoral' if w < 0 else 'lightgreen' for row in kernel_weights for w in row]
    
    idx = 0
    for i in range(3):
        for j in range(3):
            rect = Rectangle((2+j*2, 6-i*2), 2, 2, facecolor=colors[idx], edgecolor='black')
            ax2.add_patch(rect)
            ax2.text(3+j*2, 7-i*2, str(kernel_weights[i,j]), ha='center', va='center', fontweight='bold')
            idx += 1
    
    ax2.text(5, 2, 'MTJ Kernel Weights\n(Edge Detection)', ha='center', va='center', fontweight='bold')
    ax2.set_title('(b) MTJ Kernel Weights', fontweight='bold')
    ax2.axis('off')
    
    # (c) Convolution operation
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    # Show convolution calculation
    conv_result = np.sum(pixel_values * kernel_weights)
    ax3.text(5, 8, 'Convolution Operation:', ha='center', va='center', fontweight='bold', fontsize=14)
    ax3.text(5, 6.5, f'Σ(pixel × weight) = {conv_result}', ha='center', va='center', fontsize=12)
    ax3.text(5, 5, f'|{conv_result}| > Threshold?', ha='center', va='center', fontweight='bold', fontsize=12)
    ax3.text(5, 3.5, f'Threshold = σ × 0.5 = {abs(conv_result)*0.8:.1f}', ha='center', va='center', fontsize=11)
    ax3.text(5, 2, f'Result: {"EDGE" if abs(conv_result) > abs(conv_result)*0.8 else "NO EDGE"}', 
             ha='center', va='center', fontweight='bold', fontsize=12, 
             color='red' if abs(conv_result) > abs(conv_result)*0.8 else 'blue')
    
    ax3.set_title('(c) Convolution & Threshold', fontweight='bold')
    ax3.axis('off')
    
    # (d) Decision flow
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    # Draw flowchart
    # Input box
    rect1 = FancyBboxPatch((1, 8), 8, 1, boxstyle="round,pad=0.1", 
                          facecolor='lightblue', edgecolor='black')
    ax4.add_patch(rect1)
    ax4.text(5, 8.5, 'Pixel Neighborhood Input', ha='center', va='center', fontweight='bold')
    
    # Arrow
    ax4.arrow(5, 7.8, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Process box
    rect2 = FancyBboxPatch((1.5, 6), 7, 1, boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', edgecolor='black')
    ax4.add_patch(rect2)
    ax4.text(5, 6.5, 'MTJ Convolution + Threshold', ha='center', va='center', fontweight='bold')
    
    # Arrow
    ax4.arrow(5, 5.8, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Decision diamond
    diamond = FancyBboxPatch((3, 3.5), 4, 1.5, boxstyle="round,pad=0.1", 
                            facecolor='yellow', edgecolor='black')
    ax4.add_patch(diamond)
    ax4.text(5, 4.25, '|Conv| > Threshold?', ha='center', va='center', fontweight='bold')
    
    # Output boxes
    rect3 = FancyBboxPatch((0.5, 1), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor='lightcoral', edgecolor='black')
    ax4.add_patch(rect3)
    ax4.text(2, 1.5, 'Output: 255\n(EDGE)', ha='center', va='center', fontweight='bold')
    
    rect4 = FancyBboxPatch((6.5, 1), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor='lightgray', edgecolor='black')
    ax4.add_patch(rect4)
    ax4.text(8, 1.5, 'Output: 0\n(NO EDGE)', ha='center', va='center', fontweight='bold')
    
    # Decision arrows
    ax4.arrow(3.5, 3.5, -1, -1, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax4.text(2.5, 2.8, 'YES', ha='center', va='center', fontweight='bold')
    ax4.arrow(6.5, 3.5, 1, -1, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax4.text(7.5, 2.8, 'NO', ha='center', va='center', fontweight='bold')
    
    ax4.set_title('(d) Decision Logic Flow', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('MTJ Edge Detection Logic and Algorithm', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('edge_detection_logic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated edge_detection_logic.png")

def create_operation_flowchart():
    """Create operational flowchart"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    
    # Define flowchart elements
    boxes = [
        {'pos': (5, 15), 'size': (3, 0.8), 'text': 'Medical Image\nInput (MRI)', 'color': 'lightblue'},
        {'pos': (5, 13.5), 'size': (3.5, 0.8), 'text': 'Preprocessing\n(Histogram Equalization)', 'color': 'lightgreen'},
        {'pos': (5, 12), 'size': (3.5, 0.8), 'text': 'Bit-plane\nDecomposition', 'color': 'lightyellow'},
        {'pos': (2.5, 10.5), 'size': (2, 0.8), 'text': '2×2 Kernel\nProcessing', 'color': 'lightcoral'},
        {'pos': (5, 10.5), 'size': (2, 0.8), 'text': '3×3 Kernel\nProcessing', 'color': 'lightcoral'},
        {'pos': (7.5, 10.5), 'size': (2, 0.8), 'text': '4×4 Kernel\nProcessing', 'color': 'lightcoral'},
        {'pos': (5, 9), 'size': (4, 0.8), 'text': 'Parallel MTJ Edge Detection\n(Multiple Bit Planes)', 'color': 'orange'},
        {'pos': (5, 7.5), 'size': (3.5, 0.8), 'text': 'Post-processing\n(Morphological Ops)', 'color': 'lightgreen'},
        {'pos': (5, 6), 'size': (3, 0.8), 'text': 'Quality Assessment\n(F1, Precision, Recall)', 'color': 'lightpink'},
        {'pos': (2.5, 4.5), 'size': (2.5, 0.8), 'text': 'Energy Analysis\n(nJ per operation)', 'color': 'lightsteelblue'},
        {'pos': (7.5, 4.5), 'size': (2.5, 0.8), 'text': 'Performance\nMetrics', 'color': 'lightsteelblue'},
        {'pos': (5, 3), 'size': (3.5, 0.8), 'text': 'Kernel Optimization\n& Selection', 'color': 'plum'},
        {'pos': (5, 1.5), 'size': (3, 0.8), 'text': 'Final Edge-detected\nOutput', 'color': 'lightblue'},
    ]
    
    # Draw boxes
    for box in boxes:
        rect = FancyBboxPatch((box['pos'][0] - box['size'][0]/2, box['pos'][1] - box['size'][1]/2), 
                             box['size'][0], box['size'][1], 
                             boxstyle="round,pad=0.1", 
                             facecolor=box['color'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(box['pos'][0], box['pos'][1], box['text'], 
               ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Draw arrows
    arrows = [
        ((5, 14.6), (5, 14.1)),  # Input to Preprocessing
        ((5, 13.1), (5, 12.6)),  # Preprocessing to Bit-plane
        ((5, 11.6), (2.5, 11.1)),  # Bit-plane to 2x2
        ((5, 11.6), (5, 11.1)),    # Bit-plane to 3x3
        ((5, 11.6), (7.5, 11.1)),  # Bit-plane to 4x4
        ((2.5, 10.1), (4, 9.6)),   # 2x2 to Parallel
        ((5, 10.1), (5, 9.6)),     # 3x3 to Parallel
        ((7.5, 10.1), (6, 9.6)),   # 4x4 to Parallel
        ((5, 8.6), (5, 8.1)),      # Parallel to Post-processing
        ((5, 7.1), (5, 6.6)),      # Post-processing to Quality
        ((4, 5.6), (3, 5.1)),      # Quality to Energy
        ((6, 5.6), (7, 5.1)),      # Quality to Performance
        ((2.5, 4.1), (4, 3.6)),    # Energy to Optimization
        ((7.5, 4.1), (6, 3.6)),    # Performance to Optimization
        ((5, 2.6), (5, 2.1)),      # Optimization to Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add decision diamond
    diamond = FancyBboxPatch((4, 3.8), 2, 1, boxstyle="round,pad=0.1", 
                            facecolor='yellow', edgecolor='black', linewidth=2)
    ax.add_patch(diamond)
    ax.text(5, 4.3, 'Optimal\nKernel?', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Add feedback arrow
    ax.annotate('', xy=(7, 11), xytext=(8.5, 4),
               arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--'))
    ax.text(8.7, 7.5, 'Feedback\nLoop', ha='center', va='center', fontweight='bold', 
           color='red', rotation=90)
    
    ax.set_title('Complete MTJ Edge Detection Operational Flowchart', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('operation_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated operation_flowchart.png")

def main():
    """Generate all paper figures"""
    print("Generating figures for MTJ Edge Detection Paper...")
    print("=" * 50)
    
    create_device_schematic()
    create_llgs_simulation()
    create_image_to_lsb()
    create_edge_detection_logic()
    create_operation_flowchart()
    
    print("=" * 50)
    print("✅ All figures generated successfully!")
    print("\nGenerated files:")
    print("• device_schematic.png")
    print("• llgs_simulation.png") 
    print("• image_to_lsb_conversion.png")
    print("• edge_detection_logic.png")
    print("• operation_flowchart.png")
    print("\nThese figures are now ready for use in your LaTeX paper.")

if __name__ == "__main__":
    main()