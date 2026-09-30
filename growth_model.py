import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def logistic_growth(t, r, k, n0):
    """
    Computes cell population over time using the logistic growth model.
    t: Time
    r: Specific growth rate
    k: Carrying capacity (maximum population)
    n0: Initial population size
    """
    return k / (1 + ((k - n0) / n0) * np.exp(-r * t))

class BioreactorGrowthModeler:
    """
    Bioengineering tool to analyze microbial growth kinetics from Optical Density (OD600) 
    or cell count data. Extracts max growth rate and generation time.
    """

    def __init__(self, time_data: np.ndarray, cell_density_data: np.ndarray):
        self.time = np.array(time_data, dtype=float)
        self.density = np.array(cell_density_data, dtype=float)
        
        if len(self.time) != len(self.density):
            raise ValueError("Time and density arrays must be the same length.")

    def fit_growth_model(self) -> dict:
        """Fits experimental data to the logistic growth model to find r, K, and N0."""
        # Initial guesses: r=0.5, K=max density, N0=min density
        p0 = [0.5, np.max(self.density), np.min(self.density)]
        
        # Bounds: growth rate > 0, K > 0, N0 > 0
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        popt, pcov = curve_fit(logistic_growth, self.time, self.density, p0=p0, bounds=bounds)
        
        r, k, n0 = popt
        r_err, k_err, n0_err = np.sqrt(np.diag(pcov))
        
        # Calculate doubling time (generation time)
        doubling_time = np.log(2) / r
        
        return {
            'growth_rate': (r, r_err),
            'carrying_capacity': (k, k_err),
            'initial_pop': (n0, n0_err),
            'doubling_time': doubling_time
        }

    def generate_growth_plot(self, output_filename: str = 'growth_curve.png'):
        """Plots the experimental growth data against the fitted logistic model."""
        params = self.fit_growth_model()
        r, _ = params['growth_rate']
        k, _ = params['carrying_capacity']
        n0, _ = params['initial_pop']
        
        time_smooth = np.linspace(np.min(self.time), np.max(self.time) * 1.1, 200)
        density_smooth = logistic_growth(time_smooth, r, k, n0)

        plt.figure(figsize=(8, 5))
        
        # Plot experimental data
        plt.scatter(self.time, self.density, color='#d95f02', s=50, label='Experimental Data (OD600)', zorder=5)
        
        # Plot fitted curve
        plt.plot(time_smooth, density_smooth, color='#2b5c8f', linewidth=2, 
                 label=f'Logistic Fit (r={r:.3f}, K={k:.2f})')
        
        # Asymptote for carrying capacity
        plt.axhline(k, color='gray', linestyle='--', alpha=0.7, label='Carrying Capacity (K)')
        
        plt.title('Bioreactor Bacterial Growth Kinetics', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Hours)')
        plt.ylabel('Cell Density (OD600)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"[+] Growth curve plot successfully saved to {output_filename}")


if __name__ == '__main__':
    # Experimental mock data for a bacterial batch culture
    # Time in hours, Cell density as Optical Density (OD600)
    time_points = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    od_readings = np.array([0.05, 0.08, 0.15, 0.35, 0.70, 1.20, 1.65, 1.85, 1.95, 1.98, 2.00])

    print("=== BIOREACTOR GROWTH MODELER EXECUTING ===")
    modeler = BioreactorGrowthModeler(time_points, od_readings)

    # Extract kinetics parameters
    results = modeler.fit_growth_model()
    r, r_err = results['growth_rate']
    k, k_err = results['carrying_capacity']
    n0, n0_err = results['initial_pop']
    td = results['doubling_time']

    print("\n[Growth Kinetics Results]")
    print(f"  Specific Growth Rate (r) : {r:.4f} ± {r_err:.4f} hr^-1")
    print(f"  Carrying Capacity (K)    : {k:.4f} ± {k_err:.4f} OD600")
    print(f"  Initial Density (N0)     : {n0:.4f} ± {n0_err:.4f} OD600")
    print(f"  Generation/Doubling Time : {td:.4f} Hours")

    # Generate visual report
    print("\nGenerating graph...")
    modeler.generate_growth_plot()
    print("\n[✓] Bioprocess kinetic modeling complete.")