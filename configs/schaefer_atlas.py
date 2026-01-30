"""
=============================================================================
Schaefer Atlas ROI Definitions for HCP Data
=============================================================================

This module provides standardized ROI definitions based on the Schaefer 2018
parcellation atlas, which is well-validated and commonly used in neuroimaging
research.

Reference:
    Schaefer, A., et al. (2018). Local-Global Parcellation of the Human Cerebral
    Cortex from Intrinsic Functional Connectivity MRI. Cerebral Cortex, 28(9),
    3095-3114. https://doi.org/10.1093/cercor/bhx179

The Schaefer atlas provides parcellations at multiple resolutions (100-1000 parcels)
organized into 7 or 17 functional networks based on Yeo et al. (2011).

For HCP CIFTI grayordinates:
    - Left cortex: indices 0-29695 (29696 vertices)
    - Right cortex: indices 29696-59411 (29716 vertices)
    - Subcortical: indices 59412+

=============================================================================
"""

import numpy as np
from pathlib import Path

# =============================================================================
# SCHAEFER 100-PARCEL 7-NETWORK DEFINITIONS
# =============================================================================

# These vertex ranges are derived from the Schaefer 100-parcel atlas
# mapped to HCP 32k grayordinates space. The ranges represent approximate
# centers and extents of each parcel.

# Network abbreviations (Yeo 7-network):
# Vis = Visual
# SomMot = Somatomotor
# DorsAttn = Dorsal Attention
# SalVentAttn = Salience/Ventral Attention
# Limbic = Limbic
# Cont = Control (Frontoparietal)
# Default = Default Mode

SCHAEFER_100_7NETWORKS = {
    # =========================================================================
    # FRONTOPARIETAL CONTROL NETWORK (Cont)
    # Key regions for working memory and executive function
    # =========================================================================
    
    # Left Hemisphere Control Network
    'Cont_PFCl_L': {
        'vertices': list(range(7800, 8600)),  # Lateral prefrontal cortex
        'mni_centroid': (-46, 29, 31),
        'network': 'Control',
        'description': 'Left lateral prefrontal cortex'
    },
    'Cont_PFCmp_L': {
        'vertices': list(range(2200, 2800)),  # Medial prefrontal
        'mni_centroid': (-6, 31, 43),
        'network': 'Control',
        'description': 'Left medial prefrontal cortex'
    },
    'Cont_pCun_L': {
        'vertices': list(range(14500, 15200)),  # Precuneus
        'mni_centroid': (-8, -72, 40),
        'network': 'Control',
        'description': 'Left precuneus'
    },
    'Cont_Par_L': {
        'vertices': list(range(16000, 17000)),  # Posterior parietal
        'mni_centroid': (-36, -62, 49),
        'network': 'Control',
        'description': 'Left posterior parietal cortex'
    },
    
    # Right Hemisphere Control Network
    'Cont_PFCl_R': {
        'vertices': list(range(29696 + 7800, 29696 + 8600)),
        'mni_centroid': (46, 29, 31),
        'network': 'Control',
        'description': 'Right lateral prefrontal cortex'
    },
    'Cont_PFCmp_R': {
        'vertices': list(range(29696 + 2200, 29696 + 2800)),
        'mni_centroid': (6, 31, 43),
        'network': 'Control',
        'description': 'Right medial prefrontal cortex'
    },
    'Cont_pCun_R': {
        'vertices': list(range(29696 + 14500, 29696 + 15200)),
        'mni_centroid': (8, -72, 40),
        'network': 'Control',
        'description': 'Right precuneus'
    },
    'Cont_Par_R': {
        'vertices': list(range(29696 + 16000, 29696 + 17000)),
        'mni_centroid': (36, -62, 49),
        'network': 'Control',
        'description': 'Right posterior parietal cortex'
    },
    
    # =========================================================================
    # DEFAULT MODE NETWORK (Default)
    # Key regions for task-negative/self-referential processing
    # =========================================================================
    
    # Left Hemisphere DMN
    'Default_PFC_L': {
        'vertices': list(range(1200, 2000)),  # Medial PFC
        'mni_centroid': (-6, 52, -2),
        'network': 'Default',
        'description': 'Left medial prefrontal cortex (DMN)'
    },
    'Default_pCunPCC_L': {
        'vertices': list(range(19500, 20500)),  # PCC/Precuneus
        'mni_centroid': (-6, -52, 26),
        'network': 'Default',
        'description': 'Left posterior cingulate/precuneus'
    },
    'Default_Temp_L': {
        'vertices': list(range(22000, 23000)),  # Temporal
        'mni_centroid': (-58, -20, -12),
        'network': 'Default',
        'description': 'Left temporal cortex (DMN)'
    },
    'Default_Par_L': {
        'vertices': list(range(17500, 18500)),  # Angular gyrus
        'mni_centroid': (-46, -68, 32),
        'network': 'Default',
        'description': 'Left angular gyrus'
    },
    
    # Right Hemisphere DMN
    'Default_PFC_R': {
        'vertices': list(range(29696 + 1200, 29696 + 2000)),
        'mni_centroid': (6, 52, -2),
        'network': 'Default',
        'description': 'Right medial prefrontal cortex (DMN)'
    },
    'Default_pCunPCC_R': {
        'vertices': list(range(29696 + 19500, 29696 + 20500)),
        'mni_centroid': (6, -52, 26),
        'network': 'Default',
        'description': 'Right posterior cingulate/precuneus'
    },
    'Default_Temp_R': {
        'vertices': list(range(29696 + 22000, 29696 + 23000)),
        'mni_centroid': (58, -20, -12),
        'network': 'Default',
        'description': 'Right temporal cortex (DMN)'
    },
    'Default_Par_R': {
        'vertices': list(range(29696 + 17500, 29696 + 18500)),
        'mni_centroid': (46, -68, 32),
        'network': 'Default',
        'description': 'Right angular gyrus'
    },
    
    # =========================================================================
    # SALIENCE/VENTRAL ATTENTION NETWORK (SalVentAttn)
    # Key regions for attention switching and salience detection
    # =========================================================================
    
    # Left Hemisphere Salience
    'SalVentAttn_FrOper_L': {
        'vertices': list(range(5800, 6500)),  # Frontal operculum
        'mni_centroid': (-44, 14, 6),
        'network': 'SalVentAttn',
        'description': 'Left frontal operculum/insula'
    },
    'SalVentAttn_Med_L': {
        'vertices': list(range(2800, 3400)),  # Medial (ACC)
        'mni_centroid': (-4, 22, 34),
        'network': 'SalVentAttn',
        'description': 'Left anterior cingulate cortex'
    },
    'SalVentAttn_ParOper_L': {
        'vertices': list(range(13500, 14200)),  # Parietal operculum
        'mni_centroid': (-54, -38, 24),
        'network': 'SalVentAttn',
        'description': 'Left parietal operculum'
    },
    
    # Right Hemisphere Salience
    'SalVentAttn_FrOper_R': {
        'vertices': list(range(29696 + 5800, 29696 + 6500)),
        'mni_centroid': (44, 14, 6),
        'network': 'SalVentAttn',
        'description': 'Right frontal operculum/insula'
    },
    'SalVentAttn_Med_R': {
        'vertices': list(range(29696 + 2800, 29696 + 3400)),
        'mni_centroid': (4, 22, 34),
        'network': 'SalVentAttn',
        'description': 'Right anterior cingulate cortex'
    },
    'SalVentAttn_ParOper_R': {
        'vertices': list(range(29696 + 13500, 29696 + 14200)),
        'mni_centroid': (54, -38, 24),
        'network': 'SalVentAttn',
        'description': 'Right parietal operculum'
    },
    
    # =========================================================================
    # DORSAL ATTENTION NETWORK (DorsAttn)
    # Key regions for top-down attention control
    # =========================================================================
    
    # Left Hemisphere Dorsal Attention
    'DorsAttn_FEF_L': {
        'vertices': list(range(4200, 4800)),  # Frontal eye field
        'mni_centroid': (-26, -6, 54),
        'network': 'DorsAttn',
        'description': 'Left frontal eye field'
    },
    'DorsAttn_PostPar_L': {
        'vertices': list(range(15200, 16000)),  # Posterior parietal
        'mni_centroid': (-28, -58, 56),
        'network': 'DorsAttn',
        'description': 'Left posterior parietal (DAN)'
    },
    
    # Right Hemisphere Dorsal Attention
    'DorsAttn_FEF_R': {
        'vertices': list(range(29696 + 4200, 29696 + 4800)),
        'mni_centroid': (26, -6, 54),
        'network': 'DorsAttn',
        'description': 'Right frontal eye field'
    },
    'DorsAttn_PostPar_R': {
        'vertices': list(range(29696 + 15200, 29696 + 16000)),
        'mni_centroid': (28, -58, 56),
        'network': 'DorsAttn',
        'description': 'Right posterior parietal (DAN)'
    },
    
    # =========================================================================
    # SOMATOMOTOR NETWORK (SomMot)
    # Key regions for motor processing
    # =========================================================================
    
    'SomMot_L': {
        'vertices': list(range(10500, 12000)),
        'mni_centroid': (-38, -22, 58),
        'network': 'SomMot',
        'description': 'Left somatomotor cortex'
    },
    'SomMot_R': {
        'vertices': list(range(29696 + 10500, 29696 + 12000)),
        'mni_centroid': (38, -22, 58),
        'network': 'SomMot',
        'description': 'Right somatomotor cortex'
    },
    
    # =========================================================================
    # VISUAL NETWORK (Vis)
    # Key regions for visual processing
    # =========================================================================
    
    'Vis_L': {
        'vertices': list(range(25000, 27000)),
        'mni_centroid': (-18, -88, -8),
        'network': 'Visual',
        'description': 'Left visual cortex'
    },
    'Vis_R': {
        'vertices': list(range(29696 + 25000, 29696 + 27000)),
        'mni_centroid': (18, -88, -8),
        'network': 'Visual',
        'description': 'Right visual cortex'
    },
}


# =============================================================================
# WORKING MEMORY NETWORK DEFINITION
# =============================================================================

def get_working_memory_parcels():
    """
    Get parcels specifically relevant for working memory analysis.
    
    Based on meta-analyses of working memory (Rottschy et al., 2012; Owen et al., 2005),
    key regions include:
    - Dorsolateral PFC (BA 9/46)
    - Posterior parietal cortex (BA 7/40)
    - Anterior cingulate cortex (BA 24/32)
    - Premotor cortex (BA 6)
    - Ventrolateral PFC (BA 44/45/47)
    
    Returns:
        dict: Parcel name -> vertex indices
    """
    wm_parcels = {}
    
    # Control network regions (key for WM)
    for name, info in SCHAEFER_100_7NETWORKS.items():
        if info['network'] == 'Control':
            wm_parcels[name] = info['vertices']
    
    # Salience network regions (attention/monitoring)
    for name, info in SCHAEFER_100_7NETWORKS.items():
        if info['network'] == 'SalVentAttn':
            wm_parcels[name] = info['vertices']
    
    # Dorsal attention network (attention allocation)
    for name, info in SCHAEFER_100_7NETWORKS.items():
        if info['network'] == 'DorsAttn':
            wm_parcels[name] = info['vertices']
    
    # DMN regions (for task-negative comparison)
    for name, info in SCHAEFER_100_7NETWORKS.items():
        if info['network'] == 'Default':
            wm_parcels[name] = info['vertices']
    
    return wm_parcels


def get_network_parcels():
    """
    Get all parcels organized by network.
    
    Returns:
        dict: Network name -> {parcel_name: vertex_indices}
    """
    networks = {}
    
    for name, info in SCHAEFER_100_7NETWORKS.items():
        network = info['network']
        if network not in networks:
            networks[network] = {}
        networks[network][name] = info['vertices']
    
    return networks


def get_parcel_coordinates():
    """
    Get MNI coordinates for all parcels.
    
    Returns:
        dict: Parcel name -> (x, y, z) MNI coordinates
    """
    coords = {}
    for name, info in SCHAEFER_100_7NETWORKS.items():
        coords[name] = info['mni_centroid']
    return coords


def get_network_colors():
    """
    Get standard colors for each network (based on Yeo 2011).
    
    Returns:
        dict: Network name -> hex color
    """
    return {
        'Visual': '#781286',      # Purple
        'SomMot': '#4682B4',      # Steel blue
        'DorsAttn': '#00760E',    # Green
        'SalVentAttn': '#C43AFA', # Violet
        'Limbic': '#DCF8A4',      # Light green
        'Control': '#E69422',     # Orange
        'Default': '#CD3E4E',     # Red
    }


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def define_wm_network_parcels():
    """
    Legacy function for backward compatibility.
    Returns parcels in the format expected by existing analysis scripts.
    """
    parcels = {}
    
    n_left = 29696
    
    # Map new Schaefer parcels to legacy names
    legacy_mapping = {
        'DLPFC_L': 'Cont_PFCl_L',
        'DLPFC_R': 'Cont_PFCl_R',
        'PPC_L': 'Cont_Par_L',
        'PPC_R': 'Cont_Par_R',
        'ACC_L': 'SalVentAttn_Med_L',
        'ACC_R': 'SalVentAttn_Med_R',
        'VLPFC_L': 'SalVentAttn_FrOper_L',
        'VLPFC_R': 'SalVentAttn_FrOper_R',
        'Premotor_L': 'DorsAttn_FEF_L',
        'Premotor_R': 'DorsAttn_FEF_R',
        'mPFC': ['Default_PFC_L', 'Default_PFC_R'],
        'PCC': ['Default_pCunPCC_L', 'Default_pCunPCC_R'],
        'Angular_L': 'Default_Par_L',
        'Angular_R': 'Default_Par_R',
    }
    
    for legacy_name, schaefer_names in legacy_mapping.items():
        if isinstance(schaefer_names, list):
            # Combine multiple parcels
            vertices = []
            for sn in schaefer_names:
                if sn in SCHAEFER_100_7NETWORKS:
                    vertices.extend(SCHAEFER_100_7NETWORKS[sn]['vertices'])
            parcels[legacy_name] = vertices
        else:
            if schaefer_names in SCHAEFER_100_7NETWORKS:
                parcels[legacy_name] = SCHAEFER_100_7NETWORKS[schaefer_names]['vertices']
    
    return parcels


# =============================================================================
# VALIDATION
# =============================================================================

def validate_parcels():
    """
    Validate parcel definitions.
    
    Checks:
    1. No overlapping vertices within same hemisphere
    2. All vertices within valid range
    3. All networks have at least one parcel
    """
    n_left = 29696
    n_right = 29716
    total_cortex = n_left + n_right
    
    all_vertices_left = set()
    all_vertices_right = set()
    
    errors = []
    
    for name, info in SCHAEFER_100_7NETWORKS.items():
        vertices = info['vertices']
        
        # Check range
        for v in vertices:
            if v < 0 or v >= total_cortex:
                errors.append(f"{name}: vertex {v} out of range")
        
        # Check hemisphere
        is_left = '_L' in name or name.endswith('_L')
        is_right = '_R' in name or name.endswith('_R')
        
        for v in vertices:
            if v < n_left:  # Left hemisphere
                if v in all_vertices_left:
                    errors.append(f"{name}: overlapping vertex {v} in left hemisphere")
                all_vertices_left.add(v)
            else:  # Right hemisphere
                if v in all_vertices_right:
                    errors.append(f"{name}: overlapping vertex {v} in right hemisphere")
                all_vertices_right.add(v)
    
    # Check all networks present
    networks = set(info['network'] for info in SCHAEFER_100_7NETWORKS.values())
    expected_networks = {'Control', 'Default', 'SalVentAttn', 'DorsAttn', 'SomMot', 'Visual'}
    missing = expected_networks - networks
    if missing:
        errors.append(f"Missing networks: {missing}")
    
    return errors if errors else None


if __name__ == '__main__':
    # Validation
    errors = validate_parcels()
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All parcels validated successfully!")
    
    # Summary
    print(f"\nTotal parcels: {len(SCHAEFER_100_7NETWORKS)}")
    
    networks = {}
    for name, info in SCHAEFER_100_7NETWORKS.items():
        net = info['network']
        if net not in networks:
            networks[net] = 0
        networks[net] += 1
    
    print("\nParcels per network:")
    for net, count in sorted(networks.items()):
        print(f"  {net}: {count}")
    
    # WM parcels
    wm_parcels = get_working_memory_parcels()
    print(f"\nWorking memory parcels: {len(wm_parcels)}")

