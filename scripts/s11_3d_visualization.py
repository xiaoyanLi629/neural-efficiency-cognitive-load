#!/usr/bin/env python3
"""
=============================================================================
3D Brain fMRI Visualization - Complete Suite
=============================================================================

This module provides comprehensive 3D visualization capabilities for fMRI data:

1. STATIC VISUALIZATIONS (Nilearn-based)
   - Glass brain views
   - Multi-slice statistical maps
   - Mosaic views
   - Surface projections
   - Volume rendering

2. INTERACTIVE 3D VISUALIZATIONS (Plotly-based)
   - Network-colored voxels
   - True cube voxel representation
   - Isosurface visualization
   - Anatomical region coloring

3. ANIMATIONS (GIF)
   - Simple rotation at single time point
   - Time evolution with fixed viewing angle
   - Combined rotation + time evolution

=============================================================================
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "3d_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default subject
SUBJECT = "100206"

# fMRI data path
FMRI_PATH = DATA_ROOT / SUBJECT / "MNINonLinear" / "Results" / "tfMRI_WM_LR" / "tfMRI_WM_LR_hp0_clean_rclean_tclean.nii.gz"

# Brain parcellation path
PARCELLATION_PATH = DATA_ROOT / SUBJECT / "MNINonLinear" / "aparc+aseg.nii.gz"

# =============================================================================
# BRAIN REGION DEFINITIONS
# =============================================================================

# FreeSurfer region labels and corresponding network/colors
BRAIN_REGIONS = {
    'Frontal': {
        'labels': [1003, 1012, 1014, 1018, 1019, 1020, 1024, 1027, 1028,
                   2003, 2012, 2014, 2018, 2019, 2020, 2024, 2027, 2028],
        'color': '#3498db',
        'name': 'Frontal Lobe'
    },
    'Parietal': {
        'labels': [1008, 1025, 1029, 1031, 2008, 2025, 2029, 2031],
        'color': '#2ecc71',
        'name': 'Parietal Lobe'
    },
    'Temporal': {
        'labels': [1001, 1006, 1007, 1009, 1015, 1016, 1030, 1034,
                   2001, 2006, 2007, 2009, 2015, 2016, 2030, 2034],
        'color': '#e67e22',
        'name': 'Temporal Lobe'
    },
    'Occipital': {
        'labels': [1005, 1011, 1013, 1021, 2005, 2011, 2013, 2021],
        'color': '#9b59b6',
        'name': 'Occipital Lobe'
    },
    'Cingulate': {
        'labels': [1002, 1010, 1023, 1026, 2002, 2010, 2023, 2026],
        'color': '#e74c3c',
        'name': 'Cingulate Cortex'
    },
    'Insula': {
        'labels': [1035, 2035],
        'color': '#1abc9c',
        'name': 'Insula'
    },
    'Subcortical': {
        'labels': [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58],
        'color': '#f1c40f',
        'name': 'Subcortical'
    },
}

# Network definitions (based on MNI coordinates)
NETWORK_REGIONS_MNI = {
    'FPN': {
        'color': '#2980b9',
        'name': 'Frontoparietal Network',
        'centers': [(-46, 45, 20), (46, 45, 20), (-40, -55, 45), (40, -55, 45)]
    },
    'DMN': {
        'color': '#c0392b',
        'name': 'Default Mode Network',
        'centers': [(0, 52, -6), (0, -52, 26), (-48, -68, 36), (48, -68, 36)]
    },
    'SAL': {
        'color': '#d35400',
        'name': 'Salience Network',
        'centers': [(0, 22, 34), (-35, 18, 2), (35, 18, 2)]
    },
    'VIS': {
        'color': '#8e44ad',
        'name': 'Visual Network',
        'centers': [(0, -82, 4), (-24, -94, -4), (24, -94, -4)]
    },
    'MOT': {
        'color': '#27ae60',
        'name': 'Motor Network',
        'centers': [(-38, -22, 56), (38, -22, 56), (0, -14, 48)]
    }
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def mni_to_voxel(mni_coord, affine):
    """Convert MNI coordinates to voxel coordinates"""
    mni = np.array([mni_coord[0], mni_coord[1], mni_coord[2], 1])
    inv_affine = np.linalg.inv(affine)
    voxel = inv_affine.dot(mni)[:3]
    return tuple(int(round(v)) for v in voxel)


def assign_network_by_distance(voxel_coords, affine, networks):
    """Assign network label based on distance to network centers"""
    voxel = np.array([voxel_coords[0], voxel_coords[1], voxel_coords[2], 1])
    mni = affine.dot(voxel)[:3]
    
    min_dist = float('inf')
    assigned_network = 'Other'
    
    for network_name, network_info in networks.items():
        for center in network_info['centers']:
            dist = np.sqrt(np.sum((np.array(mni) - np.array(center))**2))
            if dist < min_dist and dist < 30:
                min_dist = dist
                assigned_network = network_name
    
    return assigned_network


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_fmri_data(fmri_path, time_point=None):
    """
    Load fMRI data
    
    Parameters:
        fmri_path: Path to fMRI file
        time_point: Selected time point (None for mean)
    
    Returns:
        img: nibabel image object
        data: 3D data array
    """
    print(f"Loading fMRI data: {fmri_path}")
    img = nib.load(fmri_path)
    data = img.get_fdata()
    
    print(f"Data dimensions: {data.shape}")
    print(f"Spatial resolution: {img.header.get_zooms()[:3]} mm")
    
    if time_point is not None:
        print(f"Selected time point: {time_point}")
        data_3d = data[:, :, :, time_point]
    else:
        print("Computing temporal mean...")
        data_3d = np.mean(data, axis=3)
    
    img_3d = nib.Nifti1Image(data_3d, img.affine, img.header)
    return img_3d, data_3d


def load_fmri_contrast(fmri_path, condition_times_0bk, condition_times_2bk, tr=0.72):
    """Compute 2-back vs 0-back activation contrast map"""
    print("Computing activation contrast (2-back > 0-back)...")
    img = nib.load(fmri_path)
    data = img.get_fdata()
    
    hrf_delay = 5
    
    def time_to_tr(times):
        return [int((t + hrf_delay) / tr) for t in times if int((t + hrf_delay) / tr) < data.shape[3]]
    
    trs_0bk = time_to_tr(condition_times_0bk)
    trs_2bk = time_to_tr(condition_times_2bk)
    
    if len(trs_0bk) > 0 and len(trs_2bk) > 0:
        mean_0bk = np.mean(data[:, :, :, trs_0bk], axis=3)
        mean_2bk = np.mean(data[:, :, :, trs_2bk], axis=3)
        contrast = mean_2bk - mean_0bk
    else:
        mid = data.shape[3] // 2
        contrast = np.mean(data[:, :, :, mid:], axis=3) - np.mean(data[:, :, :, :mid], axis=3)
    
    contrast_img = nib.Nifti1Image(contrast, img.affine, img.header)
    return contrast_img


# =============================================================================
# STATIC VISUALIZATION FUNCTIONS (Nilearn-based)
# =============================================================================

def plot_glass_brain(img, output_path, title="Glass Brain Visualization"):
    """Plot Glass Brain visualization"""
    from nilearn import plotting
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = ['x', 'y', 'z']
    titles = ['Sagittal View', 'Coronal View', 'Axial View']
    
    for ax, view, t in zip(axes, views, titles):
        plotting.plot_glass_brain(img, display_mode=view, axes=ax,
                                   colorbar=True, cmap='coolwarm', title=t)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stat_map_slices(img, output_path, title="Brain Activation"):
    """Plot statistical map with multiple slices"""
    from nilearn import plotting
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    plotting.plot_stat_map(img, display_mode='z', cut_coords=7,
                           axes=axes[0], colorbar=True, cmap='coolwarm',
                           title='Axial Slices')
    plotting.plot_stat_map(img, display_mode='x', cut_coords=7,
                           axes=axes[1], colorbar=True, cmap='coolwarm',
                           title='Sagittal Slices')
    plotting.plot_stat_map(img, display_mode='y', cut_coords=7,
                           axes=axes[2], colorbar=True, cmap='coolwarm',
                           title='Coronal Slices')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mosaic(img, output_path, title="Brain Mosaic"):
    """Plot mosaic view (multiple slices)"""
    from nilearn import plotting
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    z_coords = np.linspace(-40, 70, 20).astype(int)
    
    for ax, z in zip(axes.flat, z_coords):
        plotting.plot_stat_map(img, display_mode='z', cut_coords=[z],
                               axes=ax, colorbar=False, cmap='coolwarm',
                               annotate=False)
        ax.set_title(f'Z = {z}', fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_3d_volume_rendering(data, affine, output_path, title="3D Volume Rendering"):
    """Create simple 3D volume rendering using matplotlib"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    threshold = np.percentile(data[data > 0], 90)
    x, y, z = np.where(data > threshold)
    values = data[x, y, z]
    values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
    
    scatter = ax.scatter(x, y, z, c=values_norm, cmap='hot', alpha=0.6, s=1, marker='.')
    
    ax.set_xlabel('X (Left-Right)')
    ax.set_ylabel('Y (Posterior-Anterior)')
    ax.set_zlabel('Z (Inferior-Superior)')
    ax.set_title(title)
    plt.colorbar(scatter, label='Activation Intensity', shrink=0.6)
    ax.view_init(elev=20, azim=45)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_3d_surface(img, output_path, title="3D Surface Projection"):
    """Plot 3D cortical surface projection"""
    from nilearn import plotting, datasets, surface
    
    print("Downloading fsaverage surface template...")
    fsaverage = datasets.fetch_surf_fsaverage()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
    views = [('left', 'lateral'), ('left', 'medial'), ('right', 'lateral'), ('right', 'medial')]
    
    for ax, (hemi, view) in zip(axes.flat, views):
        if hemi == 'left':
            surf_mesh = fsaverage.pial_left
            bg_map = fsaverage.sulc_left
        else:
            surf_mesh = fsaverage.pial_right
            bg_map = fsaverage.sulc_right
        
        texture = surface.vol_to_surf(img, surf_mesh)
        plotting.plot_surf_stat_map(surf_mesh, texture, hemi=hemi, view=view,
                                     bg_map=bg_map, axes=ax, colorbar=False,
                                     cmap='coolwarm', threshold=0.1)
        ax.set_title(f'{hemi.capitalize()} {view.capitalize()}')
    
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_interactive_3d(img, output_path):
    """Create interactive 3D visualization (HTML) using nilearn"""
    from nilearn import plotting
    
    view = plotting.view_img(img, threshold='auto', cmap='coolwarm',
                             title='Interactive 3D Brain Viewer')
    view.save_as_html(str(output_path))
    print(f"Saved interactive visualization: {output_path}")


# =============================================================================
# INTERACTIVE 3D VISUALIZATION FUNCTIONS (Plotly-based)
# =============================================================================

def create_plotly_3d_basic(data, affine, output_path, title="Interactive 3D Brain"):
    """Create basic interactive 3D visualization using Plotly"""
    try:
        import plotly.graph_objects as go
        
        threshold = np.percentile(np.abs(data[~np.isnan(data)]), 95)
        pos_mask = data > threshold
        neg_mask = data < -threshold
        
        fig = go.Figure()
        
        if np.any(pos_mask):
            x, y, z = np.where(pos_mask)
            values = data[pos_mask]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                marker=dict(size=2, color=values, colorscale='Reds', opacity=0.6,
                           colorbar=dict(title='Activation', x=1.1)),
                name='Positive Activation'
            ))
        
        if np.any(neg_mask):
            x, y, z = np.where(neg_mask)
            values = data[neg_mask]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                marker=dict(size=2, color=values, colorscale='Blues_r', opacity=0.6),
                name='Negative Activation'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            width=900, height=700
        )
        
        fig.write_html(str(output_path))
        print(f"Saved Plotly visualization: {output_path}")
        
    except ImportError:
        print("Plotly not installed, skipping interactive 3D visualization")


def create_plotly_3d_network_colored(fmri_path, output_path, time_point=100):
    """Create Plotly 3D visualization colored by brain network"""
    import plotly.graph_objects as go
    
    print(f"Loading fMRI data (time point: {time_point})...")
    img = nib.load(fmri_path)
    data = img.get_fdata()[:, :, :, time_point]
    affine = img.affine
    
    threshold = np.percentile(np.abs(data[~np.isnan(data) & (data != 0)]), 95)
    pos_mask = data > threshold
    neg_mask = data < -threshold
    
    fig = go.Figure()
    
    network_voxels = {net: {'x': [], 'y': [], 'z': [], 'values': []}
                      for net in list(NETWORK_REGIONS_MNI.keys()) + ['Other']}
    
    x, y, z = np.where(pos_mask)
    for i in range(len(x)):
        voxel = (x[i], y[i], z[i])
        network = assign_network_by_distance(voxel, affine, NETWORK_REGIONS_MNI)
        network_voxels[network]['x'].append(x[i])
        network_voxels[network]['y'].append(y[i])
        network_voxels[network]['z'].append(z[i])
        network_voxels[network]['values'].append(data[voxel])
    
    for network_name, voxels in network_voxels.items():
        if len(voxels['x']) == 0:
            continue
        
        if network_name in NETWORK_REGIONS_MNI:
            color = NETWORK_REGIONS_MNI[network_name]['color']
            name = NETWORK_REGIONS_MNI[network_name]['name']
        else:
            color = '#95a5a6'
            name = 'Other regions'
        
        fig.add_trace(go.Scatter3d(
            x=voxels['x'], y=voxels['y'], z=voxels['z'], mode='markers',
            marker=dict(size=3, color=color, opacity=0.7, symbol='square',
                       line=dict(width=0.5, color='white')),
            name=f'{name} ({len(voxels["x"])} voxels)',
            text=[f'Value: {v:.4f}' for v in voxels['values']],
            hoverinfo='text+name'
        ))
    
    x_neg, y_neg, z_neg = np.where(neg_mask)
    if len(x_neg) > 0:
        fig.add_trace(go.Scatter3d(
            x=x_neg, y=y_neg, z=z_neg, mode='markers',
            marker=dict(size=3, color='#3498db', opacity=0.5, symbol='square'),
            name=f'Deactivation ({len(x_neg)} voxels)'
        ))
    
    fig.update_layout(
        title=dict(text='<b>3D Brain fMRI - Network Colored</b>', x=0.5),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        width=1100, height=800
    )
    
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def create_plotly_3d_voxel_cubes(fmri_path, output_path, time_point=100):
    """Create true cube voxel visualization using Mesh3d"""
    import plotly.graph_objects as go
    
    print(f"Loading fMRI data...")
    img = nib.load(fmri_path)
    data = img.get_fdata()[:, :, :, time_point]
    affine = img.affine
    
    threshold = np.percentile(np.abs(data[~np.isnan(data) & (data != 0)]), 98)
    pos_mask = data > threshold
    x, y, z = np.where(pos_mask)
    
    print(f"Displaying {len(x)} high-activation voxels (cubes)")
    
    cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
    ])
    
    all_vertices, all_faces, all_colors = [], [], []
    
    for i in range(min(len(x), 500)):
        vx, vy, vz = x[i], y[i], z[i]
        network = assign_network_by_distance((vx, vy, vz), affine, NETWORK_REGIONS_MNI)
        color = NETWORK_REGIONS_MNI.get(network, {}).get('color', '#95a5a6')
        
        vertex_offset = len(all_vertices)
        for v in cube_vertices:
            all_vertices.append([vx + v[0], vy + v[1], vz + v[2]])
        
        for f in cube_faces:
            all_faces.append([f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset])
            all_colors.append(color)
    
    all_vertices = np.array(all_vertices)
    all_faces = np.array(all_faces)
    
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=all_vertices[:, 0], y=all_vertices[:, 1], z=all_vertices[:, 2],
        i=all_faces[:, 0], j=all_faces[:, 1], k=all_faces[:, 2],
        facecolor=all_colors, opacity=0.8, flatshading=True, name='Activated Voxels'
    ))
    
    fig.update_layout(
        title=dict(text='<b>3D Voxel Cube Visualization</b>', x=0.5),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1000, height=800
    )
    
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def create_plotly_3d_isosurface(fmri_path, output_path, time_point=100):
    """Create isosurface visualization"""
    import plotly.graph_objects as go
    from scipy.ndimage import gaussian_filter
    
    print(f"Loading data for isosurface...")
    img = nib.load(fmri_path)
    data = img.get_fdata()[:, :, :, time_point]
    data_smooth = gaussian_filter(data, sigma=1)
    
    X, Y, Z = np.mgrid[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]]
    threshold_high = np.percentile(data_smooth[data_smooth > 0], 90)
    threshold_low = np.percentile(data_smooth[data_smooth < 0], 10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=data_smooth.flatten(),
        isomin=threshold_high, isomax=data_smooth.max(), surface_count=3,
        colorscale='Reds', opacity=0.6,
        caps=dict(x_show=False, y_show=False, z_show=False), name='Positive'
    ))
    
    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=(-data_smooth).flatten(),
        isomin=-threshold_low, isomax=(-data_smooth).max(), surface_count=2,
        colorscale='Blues', opacity=0.4,
        caps=dict(x_show=False, y_show=False, z_show=False), name='Negative'
    ))
    
    fig.update_layout(
        title=dict(text='<b>3D Isosurface Visualization</b>', x=0.5),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1000, height=800
    )
    
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def create_plotly_3d_anatomical(fmri_path, output_path, time_point=100):
    """Create 3D visualization colored by anatomical parcellation"""
    import plotly.graph_objects as go
    
    print("Loading fMRI data...")
    img = nib.load(fmri_path)
    data = img.get_fdata()[:, :, :, time_point]
    affine = img.affine
    
    parcellation_data = None
    if PARCELLATION_PATH.exists():
        print("Loading brain parcellation...")
        parc_img = nib.load(PARCELLATION_PATH)
        parcellation_data = parc_img.get_fdata()
    
    threshold = np.percentile(np.abs(data[~np.isnan(data) & (data != 0)]), 95)
    activated_mask = np.abs(data) > threshold
    x, y, z = np.where(activated_mask)
    values = data[activated_mask]
    
    fig = go.Figure()
    
    if parcellation_data is not None:
        region_voxels = {}
        
        for i in range(len(x)):
            vx, vy, vz = x[i], y[i], z[i]
            if vx < parcellation_data.shape[0] and vy < parcellation_data.shape[1] and vz < parcellation_data.shape[2]:
                label = int(parcellation_data[vx, vy, vz])
            else:
                label = 0
            
            region_name, region_color = 'Other', '#7f8c8d'
            for region, info in BRAIN_REGIONS.items():
                if label in info['labels']:
                    region_name, region_color = region, info['color']
                    break
            
            if region_name not in region_voxels:
                region_voxels[region_name] = {
                    'x': [], 'y': [], 'z': [], 'values': [],
                    'color': region_color,
                    'full_name': BRAIN_REGIONS.get(region_name, {}).get('name', 'Other')
                }
            
            region_voxels[region_name]['x'].append(vx)
            region_voxels[region_name]['y'].append(vy)
            region_voxels[region_name]['z'].append(vz)
            region_voxels[region_name]['values'].append(values[i])
        
        for region_name, voxels in region_voxels.items():
            if len(voxels['x']) == 0:
                continue
            
            fig.add_trace(go.Scatter3d(
                x=voxels['x'], y=voxels['y'], z=voxels['z'], mode='markers',
                marker=dict(size=4, color=voxels['color'], opacity=0.7, symbol='square'),
                name=f"{voxels['full_name']} ({len(voxels['x'])})"
            ))
    else:
        print("Parcellation not found, using network-based coloring...")
        network_voxels = {net: {'x': [], 'y': [], 'z': [], 'values': []}
                         for net in list(NETWORK_REGIONS_MNI.keys()) + ['Other']}
        
        for i in range(len(x)):
            network = assign_network_by_distance((x[i], y[i], z[i]), affine, NETWORK_REGIONS_MNI)
            network_voxels[network]['x'].append(x[i])
            network_voxels[network]['y'].append(y[i])
            network_voxels[network]['z'].append(z[i])
            network_voxels[network]['values'].append(values[i])
        
        for network_name, voxels in network_voxels.items():
            if len(voxels['x']) == 0:
                continue
            
            color = NETWORK_REGIONS_MNI.get(network_name, {}).get('color', '#95a5a6')
            name = NETWORK_REGIONS_MNI.get(network_name, {}).get('name', 'Other')
            
            fig.add_trace(go.Scatter3d(
                x=voxels['x'], y=voxels['y'], z=voxels['z'], mode='markers',
                marker=dict(size=4, color=color, opacity=0.7, symbol='square'),
                name=f"{name} ({len(voxels['x'])})"
            ))
    
    fig.update_layout(
        title=dict(text='<b>3D Brain Activation by Region</b>', x=0.5),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        width=1100, height=800
    )
    
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def create_network_legend():
    """Create network color legend image"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    y_pos = 0
    for network_name, info in NETWORK_REGIONS_MNI.items():
        ax.barh(y_pos, 1, color=info['color'], height=0.6)
        ax.text(1.1, y_pos, f"{info['name']}", va='center', fontsize=11)
        y_pos += 1
    
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, y_pos - 0.5)
    ax.axis('off')
    ax.set_title('Brain Network Color Legend', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'network_color_legend.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'network_color_legend.png'}")


# =============================================================================
# ANIMATION FUNCTIONS (GIF)
# =============================================================================

def create_dual_isosurface_frame(data, ax, azim, elev, time_label):
    """Create frame with both positive and negative activation isosurfaces"""
    from scipy.ndimage import gaussian_filter
    from skimage import measure
    
    ax.clear()
    data_smooth = gaussian_filter(data, sigma=1.5)
    
    pos_threshold = np.percentile(data_smooth[data_smooth > 0], 88) if np.any(data_smooth > 0) else 0
    neg_threshold = np.percentile(data_smooth[data_smooth < 0], 12) if np.any(data_smooth < 0) else 0
    
    try:
        if pos_threshold > 0:
            verts_pos, faces_pos, _, _ = measure.marching_cubes(data_smooth, level=pos_threshold, allow_degenerate=False)
            ax.plot_trisurf(verts_pos[:, 0], verts_pos[:, 1], verts_pos[:, 2],
                           triangles=faces_pos, color='#e74c3c', alpha=0.7, shade=True, edgecolor='none')
    except:
        pass
    
    try:
        if neg_threshold < 0:
            verts_neg, faces_neg, _, _ = measure.marching_cubes(-data_smooth, level=-neg_threshold, allow_degenerate=False)
            ax.plot_trisurf(verts_neg[:, 0], verts_neg[:, 1], verts_neg[:, 2],
                           triangles=faces_neg, color='#3498db', alpha=0.5, shade=True, edgecolor='none')
    except:
        pass
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(0, data.shape[1])
    ax.set_zlim(0, data.shape[2])
    ax.set_title(f'Brain Activity - {time_label}', fontsize=12, fontweight='bold')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    return ax


def create_rotation_gif(fmri_path, output_path, time_point=100, n_frames=72, fps=15):
    """Create a simple rotating GIF at a single time point"""
    import imageio
    
    print(f"Loading fMRI data at time point {time_point}...")
    img = nib.load(fmri_path)
    data_3d = img.get_fdata()[:, :, :, time_point]
    
    print(f"Creating {n_frames} rotation frames...")
    
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    frames = []
    time_sec = time_point * 0.72
    
    for i in range(n_frames):
        azim = (i / n_frames) * 360
        elev = 20 + 10 * np.sin(2 * np.pi * i / n_frames)
        
        create_dual_isosurface_frame(data_3d, ax, azim, elev, f"t = {time_sec:.1f}s")
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        if (i + 1) % 18 == 0:
            print(f"  Generated {i + 1}/{n_frames} frames...")
    
    plt.close(fig)
    
    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved! File size: {file_size:.1f} MB")
    return output_path


def create_time_evolution_gif(fmri_path, output_path, fps=8):
    """Create GIF showing time evolution of brain activity"""
    import imageio
    
    print("Loading fMRI data...")
    img = nib.load(fmri_path)
    data_4d = img.get_fdata()
    
    time_points = list(range(20, 350, 5))
    print(f"Creating animation with {len(time_points)} time points...")
    
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    frames = []
    
    for i, t in enumerate(time_points):
        data_3d = data_4d[:, :, :, t]
        time_sec = t * 0.72
        
        create_dual_isosurface_frame(data_3d, ax, 45, 25, f"t = {time_sec:.1f}s (TR {t})")
        
        progress = i / len(time_points)
        ax_progress = fig.add_axes([0.1, 0.02, 0.8, 0.02])
        ax_progress.barh(0, progress, color='#3498db', height=1)
        ax_progress.barh(0, 1 - progress, left=progress, color='#ecf0f1', height=1)
        ax_progress.set_xlim(0, 1)
        ax_progress.axis('off')
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        ax_progress.remove()
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{len(time_points)} frames...")
    
    plt.close(fig)
    
    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved! File size: {file_size:.1f} MB")
    return output_path


def create_combined_animation_gif(fmri_path, output_path, fps=12):
    """Create combined animation: rotation + time evolution"""
    import imageio
    
    print("Loading fMRI data...")
    img = nib.load(fmri_path)
    data_4d = img.get_fdata()
    
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    frames = []
    
    # Part 1: Rotate at baseline
    print("Part 1: Baseline rotation...")
    data_baseline = data_4d[:, :, :, 20]
    for i in range(36):
        azim = i * 10
        create_dual_isosurface_frame(data_baseline, ax, azim, 20, "Baseline (t=14.4s)")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    # Part 2: Time evolution
    print("Part 2: Time evolution...")
    time_points = list(range(20, 300, 8))
    for t in time_points:
        data_3d = data_4d[:, :, :, t]
        time_sec = t * 0.72
        create_dual_isosurface_frame(data_3d, ax, 45, 20, f"t = {time_sec:.1f}s")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    # Part 3: Rotate at task peak
    print("Part 3: Task peak rotation...")
    data_peak = data_4d[:, :, :, 100]
    for i in range(36):
        azim = i * 10
        create_dual_isosurface_frame(data_peak, ax, azim, 20, "Task Peak (t=72s)")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    plt.close(fig)
    
    print(f"Saving GIF ({len(frames)} frames) to {output_path}...")
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved! File size: {file_size:.1f} MB")
    return output_path


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_static_visualizations(fmri_path=None, output_dir=None):
    """Run all static visualization functions"""
    fmri_path = fmri_path or FMRI_PATH
    output_dir = output_dir or OUTPUT_DIR
    
    print("\n" + "=" * 70)
    print(" Static Visualizations (Nilearn)")
    print("=" * 70)
    
    time_points = {
        'baseline': 10,
        'task_peak': 100,
        'mean': None
    }
    
    for name, tp in time_points.items():
        print(f"\n--- Processing: {name} ---")
        img, data = load_fmri_data(fmri_path, time_point=tp)
        
        plot_glass_brain(img, output_dir / f"glass_brain_{name}.png",
                        title=f"Glass Brain - {name.replace('_', ' ').title()}")
        plot_stat_map_slices(img, output_dir / f"slices_{name}.png",
                            title=f"Multi-slice View - {name.replace('_', ' ').title()}")
        plot_mosaic(img, output_dir / f"mosaic_{name}.png",
                   title=f"Mosaic View - {name.replace('_', ' ').title()}")
        plot_3d_volume_rendering(data, img.affine, output_dir / f"volume_3d_{name}.png",
                                 title=f"3D Volume - {name.replace('_', ' ').title()}")
    
    # Interactive viewer
    img_mean, _ = load_fmri_data(fmri_path, time_point=None)
    plot_interactive_3d(img_mean, output_dir / "interactive_brain_viewer.html")


def run_plotly_visualizations(fmri_path=None, output_dir=None, time_point=100):
    """Run all Plotly-based 3D visualizations"""
    fmri_path = fmri_path or FMRI_PATH
    output_dir = output_dir or OUTPUT_DIR
    
    print("\n" + "=" * 70)
    print(" Interactive 3D Visualizations (Plotly)")
    print("=" * 70)
    
    create_plotly_3d_network_colored(fmri_path, output_dir / "plotly_3d_network_colored.html", time_point)
    create_plotly_3d_voxel_cubes(fmri_path, output_dir / "plotly_3d_voxel_cubes.html", time_point)
    create_plotly_3d_isosurface(fmri_path, output_dir / "plotly_3d_isosurface.html", time_point)
    create_plotly_3d_anatomical(fmri_path, output_dir / "plotly_3d_anatomical.html", time_point)
    create_network_legend()


def run_animations(fmri_path=None, output_dir=None):
    """Run all animation functions"""
    fmri_path = fmri_path or FMRI_PATH
    output_dir = output_dir or OUTPUT_DIR
    
    print("\n" + "=" * 70)
    print(" Animations (GIF)")
    print("=" * 70)
    
    create_rotation_gif(fmri_path, output_dir / "brain_rotation_3d.gif")
    create_time_evolution_gif(fmri_path, output_dir / "brain_time_evolution.gif")
    create_combined_animation_gif(fmri_path, output_dir / "brain_combined_animation.gif")


def main():
    """Main function - run all visualizations"""
    print("=" * 70)
    print(" 3D Brain fMRI Visualization - Complete Suite")
    print("=" * 70)
    
    if not FMRI_PATH.exists():
        print(f"Error: fMRI data file not found: {FMRI_PATH}")
        return
    
    # Run all visualization types
    run_static_visualizations()
    run_plotly_visualizations()
    run_animations()
    
    print("\n" + "=" * 70)
    print(" Complete! All visualizations saved to:")
    print(f" {OUTPUT_DIR}")
    print("=" * 70)
    
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

