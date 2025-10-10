"""
Postprocessing Manager

Coordinates all postprocessing tasks.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from postprocess.post_utils.data_loader import DataLoader
from postprocess.post_utils.spectrum_analyzer import SpectrumAnalyzer
from postprocess.post_utils.visualizer import Visualizer
from postprocess.post_utils.field_analyzer import FieldAnalyzer  # NEW
from postprocess.post_utils.field_exporter import FieldExporter  # NEW


class PostprocessManager:
    """Manages the entire postprocessing workflow."""
    
    def __init__(self, config, verbose=False):
        self.config = config
        self.verbose = verbose
        self.output_dir = config.get('output_dir', './results')
        
        # Initialize components
        self.data_loader = DataLoader(config, verbose)
        self.analyzer = SpectrumAnalyzer(config, verbose)
        self.visualizer = Visualizer(config, verbose)
        self.field_analyzer = FieldAnalyzer(verbose)  # NEW
        self.field_exporter = FieldExporter(self.output_dir, verbose)  # NEW
    
    def run(self):
        """Execute complete postprocessing workflow."""
        if self.verbose:
            print("\n" + "="*60)
            print("Starting Postprocessing")
            print("="*60)
        
        # Step 1: Load data
        if self.verbose:
            print("\n[1/5] Loading simulation results...")
        
        try:
            data = self.data_loader.load_simulation_results()
        except FileNotFoundError:
            print("  MAT file not found, trying text file...")
            data = self.data_loader.load_text_results()
        
        # Step 2: Analyze spectra
        if self.verbose:
            print("\n[2/5] Analyzing spectra...")
        
        analysis = self.analyzer.analyze(data)
        
        if self.verbose:
            self._print_analysis_summary(analysis)
        
        # Step 2.5: Analyze fields (NEW)
        field_analysis = []
        if 'fields' in data and data['fields']:
            if self.verbose:
                print("\n[2.5/5] Analyzing electromagnetic fields...")
            
            for field_data in data['fields']:
                field_result = self.field_analyzer.analyze_field(field_data)
                field_analysis.append(field_result)
        
        # Step 3: Create visualizations
        if self.verbose:
            print("\n[3/5] Creating visualizations...")
        
        plots = self.visualizer.create_all_plots(data)
        
        if self.verbose and plots:
            print(f"  Created {len(plots)} plot(s)")
        
        # Step 4: Export field data (NEW)
        if 'fields' in data and data['fields'] and field_analysis:
            if self.verbose:
                print("\n[4/5] Exporting field data...")
            
            # Export field analysis to JSON
            self.field_exporter.export_to_json(data['fields'], field_analysis)
            
            # Optionally export downsampled field arrays
            if self.config.get('export_field_arrays', False):
                self.field_exporter.export_field_data_arrays(data['fields'])
        
        # Step 5: Save processed data
        if self.verbose:
            step_num = "[5/5]" if 'fields' not in data or not data['fields'] else "[5/5]"
            print(f"\n{step_num} Saving processed data...")
        
        self._save_processed_data(data, analysis, field_analysis)
        
        if self.verbose:
            print("\n" + "="*60)
            print("Postprocessing Completed Successfully")
            print("="*60)
            print(f"\nResults saved in: {self.output_dir}/")
        
        return data, analysis, field_analysis
    
    def _print_analysis_summary(self, analysis):
        """Print summary of spectral analysis."""
        print("\n  Spectral Analysis Summary:")
        print("  " + "-"*50)
        
        for pol_idx, pol_analysis in enumerate(analysis):
            print(f"\n  Polarization {pol_idx + 1}:")
            
            if 'extinction_peak' in pol_analysis:
                peak = pol_analysis['extinction_peak']
                print(f"    Extinction Peak:")
                print(f"      Wavelength: {peak['wavelength']:.2f} nm")
                print(f"      Value: {peak['value']:.2e} nm²")
                print(f"      FWHM: {peak.get('fwhm', 'N/A')} nm")
            
            if 'scattering_peak' in pol_analysis:
                peak = pol_analysis['scattering_peak']
                print(f"    Scattering Peak:")
                print(f"      Wavelength: {peak['wavelength']:.2f} nm")
                print(f"      Value: {peak['value']:.2e} nm²")
    
    def _save_processed_data(self, data, analysis, field_analysis):
        """Save processed data in various formats."""
        output_formats = self.config.get('output_formats', ['txt', 'csv', 'json'])
        
        # Save analysis results
        if 'txt' in output_formats:
            self._save_txt(data, analysis, field_analysis)
        
        if 'csv' in output_formats:
            self._save_csv(data, analysis)
        
        if 'json' in output_formats:
            self._save_json(data, analysis, field_analysis)
    
    def _save_txt(self, data, analysis, field_analysis):
        """Save processed data as text file."""
        filepath = os.path.join(self.output_dir, 'simulation_processed.txt')
        
        with open(filepath, 'w') as f:
            f.write("MNPBEM Simulation Results - Processed\n")
            f.write("="*60 + "\n\n")
            
            # Write analysis summary
            f.write("SPECTRAL ANALYSIS\n")
            f.write("-"*60 + "\n\n")
            
            for pol_idx, pol_analysis in enumerate(analysis):
                f.write(f"Polarization {pol_idx + 1}:\n")
                
                if 'extinction_peak' in pol_analysis:
                    peak = pol_analysis['extinction_peak']
                    f.write(f"  Extinction Peak: {peak['wavelength']:.2f} nm\n")
                    f.write(f"  Peak Value: {peak['value']:.2e} nm²\n")
                    if 'fwhm' in peak:
                        f.write(f"  FWHM: {peak['fwhm']:.2f} nm\n")
                
                f.write("\n")
            
            # Write field analysis (NEW)
            if field_analysis:
                f.write("\nFIELD ANALYSIS\n")
                f.write("-"*60 + "\n\n")
                
                for pol_idx, field_result in enumerate(field_analysis):
                    f.write(f"Polarization {pol_idx + 1} (λ = {field_result['wavelength']:.1f} nm):\n")
                    
                    stats = field_result['enhancement_stats']
                    f.write(f"  Enhancement Statistics:\n")
                    f.write(f"    Max:       {stats['max']:.2f}\n")
                    f.write(f"    Mean:      {stats['mean']:.2f}\n")
                    f.write(f"    Median:    {stats['median']:.2f}\n")
                    f.write(f"    95th %ile: {stats['percentile_95']:.2f}\n")
                    
                    if field_result['hotspots']:
                        f.write(f"\n  Top Hotspots:\n")
                        for hotspot in field_result['hotspots'][:5]:
                            pos = hotspot['position']
                            f.write(f"    #{hotspot['rank']}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) nm "
                                  f"| E/E₀ = {hotspot['enhancement']:.2f}\n")
                    
                    f.write("\n")
            
            # Write full spectrum data
            f.write("\nFULL SPECTRUM DATA\n")
            f.write("-"*60 + "\n\n")
            f.write("Wavelength(nm)\t")
            
            n_pol = data['extinction'].shape[1]
            for i in range(n_pol):
                f.write(f"Ext_pol{i+1}\t")
            for i in range(n_pol):
                f.write(f"Sca_pol{i+1}\t")
            for i in range(n_pol):
                f.write(f"Abs_pol{i+1}")
                if i < n_pol - 1:
                    f.write("\t")
            f.write("\n")
            
            for i, wl in enumerate(data['wavelength']):
                f.write(f"{wl:.2f}\t")
                for pol in range(n_pol):
                    f.write(f"{data['extinction'][i, pol]:.6e}\t")
                for pol in range(n_pol):
                    f.write(f"{data['scattering'][i, pol]:.6e}\t")
                for pol in range(n_pol):
                    f.write(f"{data['absorption'][i, pol]:.6e}")
                    if pol < n_pol - 1:
                        f.write("\t")
                f.write("\n")
        
        if self.verbose:
            print(f"  Saved: {filepath}")
    
    def _save_csv(self, data, analysis):
        """Save processed data as CSV file."""
        import csv
        
        filepath = os.path.join(self.output_dir, 'simulation_processed.csv')
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            n_pol = data['extinction'].shape[1]
            header = ['Wavelength(nm)']
            for i in range(n_pol):
                header.append(f'Extinction_pol{i+1}')
            for i in range(n_pol):
                header.append(f'Scattering_pol{i+1}')
            for i in range(n_pol):
                header.append(f'Absorption_pol{i+1}')
            
            writer.writerow(header)
            
            # Data
            for i, wl in enumerate(data['wavelength']):
                row = [wl]
                for pol in range(n_pol):
                    row.append(data['extinction'][i, pol])
                for pol in range(n_pol):
                    row.append(data['scattering'][i, pol])
                for pol in range(n_pol):
                    row.append(data['absorption'][i, pol])
                writer.writerow(row)
        
        if self.verbose:
            print(f"  Saved: {filepath}")
    
    def _save_json(self, data, analysis, field_analysis):
        """Save processed data as JSON file."""
        import json
        
        filepath = os.path.join(self.output_dir, 'simulation_processed.json')
        
        # Prepare JSON-serializable data
        json_data = {
            'wavelength': data['wavelength'].tolist(),
            'extinction': data['extinction'].tolist(),
            'scattering': data['scattering'].tolist(),
            'absorption': data['absorption'].tolist(),
            'analysis': []
        }
        
        # Add analysis results
        for pol_analysis in analysis:
            pol_dict = {}
            for key, value in pol_analysis.items():
                if isinstance(value, dict):
                    pol_dict[key] = {k: float(v) if hasattr(v, 'item') else v
                                     for k, v in value.items()}
                else:
                    pol_dict[key] = float(value) if hasattr(value, 'item') else value
            json_data['analysis'].append(pol_dict)
        
        # Add field data summary if available
        if 'fields' in data and data['fields']:
            json_data['field_data_available'] = True
            json_data['field_wavelengths'] = [f['wavelength'] for f in data['fields']]
            
            # Add field analysis summary (NEW)
            if field_analysis:
                json_data['field_analysis_summary'] = []
                for field_result in field_analysis:
                    summary = {
                        'wavelength': field_result['wavelength'],
                        'max_enhancement': field_result['enhancement_stats']['max'],
                        'mean_enhancement': field_result['enhancement_stats']['mean'],
                        'num_hotspots': len(field_result['hotspots']),
                        'top_hotspot_position': field_result['hotspots'][0]['position'] if field_result['hotspots'] else None,
                        'top_hotspot_enhancement': field_result['hotspots'][0]['enhancement'] if field_result['hotspots'] else None
                    }
                    json_data['field_analysis_summary'].append(summary)
        else:
            json_data['field_data_available'] = False
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if self.verbose:
            print(f"  Saved: {filepath}")