"""
Köppen-Geiger Climate Classification and AOD550 Uncertainty Module

This module provides functions for:
1. Köppen-Geiger climate zone classification from lat/lon coordinates
2. Regionalized AOD550 uncertainty estimation based on climate zones

Based on Gueymard & Yang (2020) validation of MERRA-2 against AERONET.

Example
-------
>>> from KG_AOD import get_KG, get_u_AOD550_regional, KoppenGeiger
>>> zone_num = get_KG(50., 10.)  # Get KG zone for a location
>>> u = get_u_AOD550_regional(48.85, 2.35, AOD550=0.1)  # Get uncertainty
"""

import numpy as np
import pandas as pd
from PIL import Image
import io

# Default file paths
#DEFAULT_LEGEND_FILE = "/mnt/ceph/user/did/FDR4VGT/data/legend.txt"
#DEFAULT_KP_MAP_FILE = "/mnt/ceph/user/did/FDR4VGT/data/1991_2020/koppen_geiger_1p0.tif"
#DEFAULT_MAPPING_FILE= "/mnt/ceph/user/did/FDR4VGT/data/kg_zoneNumGueymard.csv"

def load_kg_map(map_file):
    """Load Köppen-Geiger classification map."""
#    if map_file is None:
#        map_file = DEFAULT_KP_MAP_FILE
    try:
        import rasterio
        with rasterio.open(map_file) as src:
            data = src.read(1)
            transform = src.transform
        return data, transform
    except ImportError:
        # Fallback to PIL for PNG
        with open(map_file, 'rb') as fp:
            img = Image.open(io.BytesIO(fp.read()))
        return img, None


def load_legend(legend_file):
    """Load legend mapping pixel values to zone names."""
#    if legend_file is None:
#        legend_file = DEFAULT_LEGEND_FILE
    legend = {}
    with open(legend_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0][:-1].isdigit():
                legend[int(parts[0][:-1])] = parts[1]
    return legend


def load_zone_mapping(mapping_file):
    """Load zone number to Gueymard zone mapping."""
#    if mapping_file is None:
#        mapping_file = DEFAULT_MAPPING_FILE
    return pd.read_csv(mapping_file)


class KoppenGeiger:
    """
    Köppen-Geiger classification handler.
    
    Example
    -------
    >>> kg = KoppenGeiger()
    >>> zone_num = kg.get_zone(lat=48.85, lon=2.35)  # Paris
    >>> zone_nums = kg.get_zone(lat=np.array([48.85, 25.0]), lon=np.array([2.35, 55.0]))
    """
    
    def __init__(self, map_file=None, legend_file=None, mapping_file=None):
        self.data, self.transform = load_kg_map(map_file)
        self.legend = load_legend(legend_file)
        self.zone_mapping = load_zone_mapping(mapping_file)
        
        # Build reverse lookup: zone_name -> zoneNum (1-32)
        self._name_to_num = dict(zip(
            self.zone_mapping['kg_zone'], 
            self.zone_mapping['zoneNum']
        ))
        
        # Get image dimensions
        if isinstance(self.data, Image.Image):
            self.width, self.height = self.data.size
        else:
            self.height, self.width = self.data.shape
    
    def _latlon_to_pixel(self, lat, lon):
        """Convert lat/lon to pixel coordinates. Supports arrays."""
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        
        if self.transform is not None:
            import rasterio
            row, col = rasterio.transform.rowcol(self.transform, lon, lat)
            return np.asarray(col).astype(int), np.asarray(row).astype(int)
        else:
            x = np.round((lon + 180) * self.width / 360 - 0.5).astype(int)
            y = np.round(-(lat - 90) * self.height / 180 - 0.5).astype(int)
            x = np.clip(x, 0, self.width - 1)
            y = np.clip(y, 0, self.height - 1)
            return x, y
    
    def get_pixel_value(self, lat, lon):
        """Get raw pixel value at location(s). Supports arrays."""
        x, y = self._latlon_to_pixel(lat, lon)
        
        if isinstance(self.data, Image.Image):
            if np.isscalar(x) or x.size == 1:
                return self.data.getpixel((int(x), int(y)))
            else:
                return np.array([self.data.getpixel((int(xi), int(yi))) 
                                for xi, yi in zip(x.flat, y.flat)]).reshape(x.shape)
        else:
            return self.data[y, x]
    
    def get_zone_name(self, lat, lon):
        """
        Get Köppen-Geiger zone name at location(s).
        
        Returns
        -------
        str or np.ndarray : Zone name(s) (e.g., 'Cfb', 'BWh', 'Ocean')
        """
        pixel_val = self.get_pixel_value(lat, lon)
        
        if np.isscalar(pixel_val):
            return self.legend.get(pixel_val, 'Ocean')
        else:
            return np.array([self.legend.get(pv, 'Ocean') 
                           for pv in pixel_val.flat]).reshape(pixel_val.shape)
    
    def get_zone(self, lat, lon, return_label=False):
        """
        Get Köppen-Geiger zone number (1-32) at location(s).
        
        Parameters
        ----------
        lat : float or np.ndarray
            Latitude in degrees (-90 to 90)
        lon : float or np.ndarray
            Longitude in degrees (-180 to 180)
        return_label : bool
            If True, also return the zone name
        
        Returns
        -------
        int or np.ndarray : Zone number 1-32 (32 = Ocean)
        or
        tuple : (Zone number(s), Zone name(s)) if return_label=True
        
        Example
        -------
        >>> kg = KoppenGeiger()
        >>> zone_num = kg.get_zone(50., 10.)  # Scalar input
        >>> lats = np.array([48.85, 25.0, 35.7])
        >>> lons = np.array([2.35, 55.0, 139.7])
        >>> zone_nums = kg.get_zone(lats, lons)  # Returns array([10, 7, ...])
        >>> zone_nums, zone_names = kg.get_zone(lats, lons, return_label=True)
        """
        zone_name = self.get_zone_name(lat, lon)
        
        if np.isscalar(zone_name) or isinstance(zone_name, str):
            zone_num = self._name_to_num.get(zone_name, 32)
        else:
            zone_num = np.array([self._name_to_num.get(zn, 32) 
                                for zn in zone_name.flat]).reshape(zone_name.shape)
        
        if return_label:
            return zone_num, zone_name
        return zone_num
    
    def get_gueymard_zone(self, lat, lon):
        """
        Get Gueymard zone number for AOD550 uncertainty lookup.
        
        Returns
        -------
        int or np.ndarray : Gueymard zone number (1-31)
        """
        zone_num = self.get_zone(lat, lon)
        
        if np.isscalar(zone_num):
            row = self.zone_mapping[self.zone_mapping['zoneNum'] == zone_num]
            return 31 if len(row) == 0 else int(row['gueymard_zoneNum'].values[0])
        else:
            result = np.zeros_like(zone_num, dtype=int)
            for i, zn in enumerate(zone_num.flat):
                row = self.zone_mapping[self.zone_mapping['zoneNum'] == zn]
                result.flat[i] = 31 if len(row) == 0 else int(row['gueymard_zoneNum'].values[0])
            return result


# Global instance (lazy initialization)
#_kg_instance = None


def get_KG(lat, lon, return_label=False):
    """
    Get Köppen-Geiger zone number (1-32) for location(s).
    
    Parameters
    ----------
    lat : float or np.ndarray
        Latitude in degrees (-90 to 90)
    lon : float or np.ndarray
        Longitude in degrees (-180 to 180)
    return_label : bool
        If True, also return the zone name
    
    Returns
    -------
    int or np.ndarray : Zone number 1-32
    
    Example
    -------
    >>> zone_num = get_KG(50., 10.)  # Scalar input
    >>> lats = np.array([48.85, 25.0, 35.7])
    >>> lons = np.array([2.35, 55.0, 139.7])
    >>> zone_nums = get_KG(lats, lons)  # Returns array([10, 7, ...])
    >>> zone_nums, zone_names = get_KG(lats, lons, return_label=True)
    """
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KoppenGeiger()
    return _kg_instance.get_zone(lat, lon, return_label=return_label)


# =============================================================================
# AOD550 Uncertainty Regionalization
# =============================================================================

# Mapping from KG zone names to uncertainty multiplier categories
KG_REGION_MULTIPLIERS = {
    # Oceanic/Polar (0.7)
    'Ocean': 0.7, 'EF': 0.7, 'ET': 0.7,
    # Dust-dominated (1.4-1.5)
    'BWh': 1.5, 'BWk': 1.5, 'BSh': 1.4, 'BSk': 1.2,
    # Biomass burning (1.3-1.4)
    'Aw': 1.4, 'As': 1.4, 'Am': 1.3, 'Af': 1.3,
    # High pollution (1.2-1.3)
    'Cwa': 1.3, 'Cfa': 1.2, 'Dwa': 1.3, 'Dwb': 1.3,
    # Baseline (1.0)
    'Cfb': 1.0, 'Cfc': 1.0, 'Csb': 1.0, 'Csa': 1.0,
    'Dfb': 1.0, 'Dfc': 1.0, 'Dfa': 1.0, 'Dsc': 1.0,
    'Dsb': 1.0, 'Dsa': 1.0, 'Dwc': 1.0, 'Dwd': 1.0,
}

DEFAULT_MULTIPLIER = 1.0


def get_aod_uncertainty_multiplier(lat, lon, _kg_instance):
    """
    Get AOD550 uncertainty multiplier based on Köppen-Geiger climate zone.
    
    Parameters
    ----------
    lat : float or np.ndarray
        Latitude in degrees (-90 to 90)
    lon : float or np.ndarray
        Longitude in degrees (-180 to 180)
    
    Returns
    -------
    float or np.ndarray : Uncertainty multiplier
    
    Example
    -------
    >>> mult = get_aod_uncertainty_multiplier(25.0, 55.0)  # Dubai -> 1.5 (BWh)
    >>> lats = np.array([48.85, 25.0, -5.0, 35.0, 80.0])
    >>> lons = np.array([2.35, 55.0, -60.0, 120.0, 0.0])
    >>> mults = get_aod_uncertainty_multiplier(lats, lons)
    """
#    global _kg_instance
#    if _kg_instance is None:
#        _kg_instance = KoppenGeiger()
    
    zone_name = _kg_instance.get_zone_name(lat, lon)
    
    if isinstance(zone_name, str) or np.isscalar(zone_name):
        return KG_REGION_MULTIPLIERS.get(zone_name, DEFAULT_MULTIPLIER)
    else:
        return np.array([KG_REGION_MULTIPLIERS.get(zn, DEFAULT_MULTIPLIER) 
                        for zn in zone_name.flat]).reshape(zone_name.shape)


def get_u_AOD550_regional(lat, lon, AOD550, base_uncertainty=0.03):
    """
    Get regionalized AOD550 uncertainty based on Köppen-Geiger climate zone.
    
    Formula: u(AOD550) = multiplier × (slope × AOD550 + base_uncertainty)
    
    Parameters
    ----------
    lat : float or np.ndarray
        Latitude in degrees (-90 to 90)
    lon : float or np.ndarray
        Longitude in degrees (-180 to 180)
    AOD550 : float or np.ndarray
        AOD value at 550nm
    base_uncertainty : float
        Baseline additive uncertainty (default 0.03)
    
    Returns
    -------
    float or np.ndarray : Uncertainty u(AOD550)
    
    Example
    -------
    >>> get_u_AOD550_regional(25.0, 55.0, 0.3)   # Dubai (dust) -> ~0.11
    >>> get_u_AOD550_regional(48.85, 2.35, 0.1)  # Paris (baseline) -> ~0.045
    >>> lats = np.array([48.85, 25.0, -5.0])
    >>> lons = np.array([2.35, 55.0, -60.0])
    >>> u = get_u_AOD550_regional(lats, lons, AOD550=np.array([0.1, 0.3, 0.2]))
    """
    slope = 0.15  # From Gueymard & Yang 2020 analysis
    multiplier = get_aod_uncertainty_multiplier(lat, lon)
    AOD550 = np.asarray(AOD550)
    return multiplier * (slope * AOD550 + base_uncertainty)