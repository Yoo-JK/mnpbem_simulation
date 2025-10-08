"""
Postprocess Manager Class

This class orchestrates the postprocessing pipeline.
"""

from pathlib import Path
from .post_utils.data_loader import DataLoader
from .post_utils.spectrum_analyzer import SpectrumAnalyzer
from .post_utils.visualizer import Visualizer


class PostprocessManager:
    """Manages the entire postprocessing pipeline."""
    
    def __init__(self, config, verbose=False):
        """
        Initialize postprocess manager.
        
        Args:
            config (dict): Configuration dictionary
            verbose (bool): Enable verbose output
        """
        self.config = config
        self.verbose = verbose
        self.data = None
        self.analysis_results = None
        
        # Initialize sub-managers
        self.data_loader = DataLoader(config, verbose)
        self.spectrum_analyzer = SpectrumAnalyzer(config, verbose)
        self.visualizer = Visualizer(config, verbose)
        
        if verbose:
            print("PostprocessManager initialized")
    
    def load_results(self):
        """Load simulation results from files."""
        if self.verbose:
            print("\n--- Loading Results ---")
        
        self.data = self.data_loader.load()
        
        if self.verbose:
            print(f"Loaded data shape:")
            print(f"  Wavelengths: {len(self.data['wavelength'])}")
            print(f"  Polarizations: {self.data['scattering'].shape[1]}")
        
        return self.data
    
    def analyze_spectrum(self):
        """Analyze the optical spectrum."""
        if self.data is None:
            raise RuntimeError("Results not loaded. Call load_results() first.")
        
        if self.verbose:
            print("\n--- Analyzing Spectrum ---")
        
        self.analysis_results = self.spectrum_analyzer.analyze(self.data)
        
        if self.verbose:
            print("Analysis complete")
            if 'peak_wavelengths' in self.analysis_results:
                print(f"Peak wavelengths: {self.analysis_results['peak_wavelengths']}")
        
        return self.analysis_results
    
    def save_processed_data(self):
        """Save processed data in various formats."""
        if self.data is None:
            raise RuntimeError("Results not loaded. Call load_results() first.")
        
        if self.verbose:
            print("\n--- Saving Processed Data ---")
        
        output_formats = self.config.get('output_formats', ['txt', 'csv'])
        
        for fmt in output_formats:
            if fmt == 'txt':
                self.data_loader.save_txt(self.data, self.analysis_results)
            elif fmt == 'csv':
                self.data_loader.save_csv(self.data, self.analysis_results)
            elif fmt == 'json':
                self.data_loader.save_json(self.data, self.analysis_results)
            
            if self.verbose:
                print(f"  Saved {fmt.upper()} format")
    
    def generate_plots(self):
        """Generate visualization plots."""
        if self.data is None:
            raise RuntimeError("Results not loaded. Call load_results() first.")
        
        if self.verbose:
            print("\n--- Generating Plots ---")
        
        # Generate spectrum plot
        self.visualizer.plot_spectrum(self.data, self.analysis_results)
        
        if self.verbose:
            print("  Generated spectrum plot")
        
        # Generate comparison plot if multiple polarizations
        if self.data['scattering'].shape[1] > 1:
            self.visualizer.plot_polarization_comparison(self.data)
            if self.verbose:
                print("  Generated polarization comparison plot")
    
    def get_summary(self):
        """Get summary of postprocessing results."""
        if self.data is None:
            return {}
        
        summary = {
            'n_wavelengths': len(self.data['wavelength']),
            'n_polarizations': self.data['scattering'].shape[1],
            'wavelength_range': [
                float(self.data['wavelength'].min()),
                float(self.data['wavelength'].max())
            ]
        }
        
        if self.analysis_results is not None:
            summary.update(self.analysis_results)
        
        return summary