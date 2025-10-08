"""
Spectrum Analyzer

Analyzes optical spectra to extract key features.
"""

import numpy as np
from scipy import signal


class SpectrumAnalyzer:
    """Analyzes optical spectrum data."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
    
    def analyze(self, data):
        """
        Perform comprehensive spectrum analysis.
        
        Args:
            data (dict): Data dictionary with wavelength, scattering, etc.
        
        Returns:
            dict: Analysis results
        """
        results = {}
        
        # Find peaks for each polarization
        peak_wavelengths, peak_values, peak_indices = self._find_peaks(
            data['wavelength'],
            data['scattering']
        )
        
        results['peak_wavelengths'] = peak_wavelengths
        results['peak_values'] = peak_values
        results['peak_indices'] = peak_indices
        
        # Calculate FWHM (Full Width at Half Maximum)
        fwhm_values = self._calculate_fwhm(
            data['wavelength'],
            data['scattering'],
            peak_indices
        )
        results['fwhm'] = fwhm_values
        
        # Calculate enhancement factors (if multiple polarizations)
        if data['n_polarizations'] > 1:
            enhancement = self._calculate_enhancement(data['scattering'])
            results['enhancement_factors'] = enhancement
        
        # Calculate average cross sections
        results['avg_scattering'] = np.mean(data['scattering'], axis=0)
        results['avg_extinction'] = np.mean(data['extinction'], axis=0)
        results['avg_absorption'] = np.mean(data['absorption'], axis=0)
        
        # Calculate max cross sections
        results['max_scattering'] = np.max(data['scattering'], axis=0)
        results['max_extinction'] = np.max(data['extinction'], axis=0)
        results['max_absorption'] = np.max(data['absorption'], axis=0)
        
        if self.verbose:
            print("Spectrum analysis complete:")
            print(f"  Peak wavelengths: {peak_wavelengths}")
            print(f"  Peak values: {peak_values}")
            print(f"  FWHM: {fwhm_values}")
        
        return results
    
    def _find_peaks(self, wavelength, cross_sections):
        """Find peak positions in spectrum."""
        n_pol = cross_sections.shape[1]
        peak_wavelengths = np.zeros(n_pol)
        peak_values = np.zeros(n_pol)
        peak_indices = np.zeros(n_pol, dtype=int)
        
        for i in range(n_pol):
            # Find peaks
            peaks, properties = signal.find_peaks(
                cross_sections[:, i],
                prominence=0.1 * np.max(cross_sections[:, i])
            )
            
            if len(peaks) > 0:
                # Get the highest peak
                max_peak_idx = peaks[np.argmax(cross_sections[peaks, i])]
                peak_indices[i] = max_peak_idx
                peak_wavelengths[i] = wavelength[max_peak_idx]
                peak_values[i] = cross_sections[max_peak_idx, i]
            else:
                # No peak found, use maximum value
                max_idx = np.argmax(cross_sections[:, i])
                peak_indices[i] = max_idx
                peak_wavelengths[i] = wavelength[max_idx]
                peak_values[i] = cross_sections[max_idx, i]
        
        return peak_wavelengths, peak_values, peak_indices
    
    def _calculate_fwhm(self, wavelength, cross_sections, peak_indices):
        """Calculate Full Width at Half Maximum."""
        n_pol = cross_sections.shape[1]
        fwhm_values = np.zeros(n_pol)
        
        for i in range(n_pol):
            peak_idx = peak_indices[i]
            peak_value = cross_sections[peak_idx, i]
            half_max = peak_value / 2
            
            # Find points where spectrum crosses half maximum
            above_half = cross_sections[:, i] > half_max
            
            # Find left edge
            left_indices = np.where(above_half[:peak_idx])[0]
            if len(left_indices) > 0:
                left_idx = left_indices[0]
            else:
                left_idx = 0
            
            # Find right edge
            right_indices = np.where(above_half[peak_idx:])[0]
            if len(right_indices) > 0:
                right_idx = peak_idx + right_indices[-1]
            else:
                right_idx = len(wavelength) - 1
            
            # Calculate FWHM
            fwhm_values[i] = wavelength[right_idx] - wavelength[left_idx]
        
        return fwhm_values
    
    def _calculate_enhancement(self, cross_sections):
        """Calculate enhancement factors between polarizations."""
        n_pol = cross_sections.shape[1]
        enhancement = {}
        
        # Calculate enhancement of first polarization relative to others
        if n_pol > 1:
            max_vals = np.max(cross_sections, axis=0)
            enhancement['pol1_vs_pol2'] = max_vals[0] / max_vals[1] if max_vals[1] > 0 else 0
            
            if n_pol > 2:
                enhancement['pol1_vs_pol3'] = max_vals[0] / max_vals[2] if max_vals[2] > 0 else 0
                enhancement['pol2_vs_pol3'] = max_vals[1] / max_vals[2] if max_vals[2] > 0 else 0
        
        return enhancement