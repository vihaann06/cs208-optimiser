import numpy as np
import matplotlib.pyplot as plt
from diffdistributions import (
    uniform_preferences,
    normal_preferences,
    mixture_normal_preferences,
    calculate_W,
    find_optimal_epsilon,
    calculate_theoretical_epsilon
)
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import norm

def run_simulation_comparison(distribution_func, name, params, sample_sizes, strict_penalty_factor=1.0):
    """
    Run simulation comparison for different sample sizes and compare with theoretical results
    """
    # Store results
    results = {
        'sample_sizes': sample_sizes,
        'sampled_epsilons': [],
        'sampled_welfares': [],
        'theoretical_epsilons': [],
        'theoretical_welfares': []
    }
    
    # Create figure for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Run simulations for each sample size
    for N in sample_sizes:
        print(f"\n=== Testing {name} distribution with N={N} ===")
        
        # Run simulation
        samples = distribution_func(**{**params, 'n_samples': N})
        epsilon_a = np.min(samples)
        epsilon_b = np.max(samples)
        sampled_epsilon, sampled_welfare = find_optimal_epsilon(epsilon_a, epsilon_b, N, strict_penalty_factor)
        
        # Calculate theoretical results
        theoretical_epsilon = calculate_theoretical_epsilon(name, params, N, strict_penalty_factor)
        theoretical_welfare = calculate_W(theoretical_epsilon, epsilon_a, epsilon_b, N, strict_penalty_factor)
        
        # Store results
        results['sampled_epsilons'].append(sampled_epsilon)
        results['sampled_welfares'].append(sampled_welfare)
        results['theoretical_epsilons'].append(theoretical_epsilon)
        results['theoretical_welfares'].append(theoretical_welfare)
        
        # Print comparison
        print(f"Sample size: {N}")
        print(f"Sampled epsilon: {sampled_epsilon:.6f}, Welfare: {sampled_welfare:.6f}")
        print(f"Theoretical epsilon: {theoretical_epsilon:.6f}, Welfare: {theoretical_welfare:.6f}")
        print(f"Relative difference in epsilon: {abs(sampled_epsilon - theoretical_epsilon)/theoretical_epsilon*100:.2f}%")
        print(f"Relative difference in welfare: {abs(sampled_welfare - theoretical_welfare)/theoretical_welfare*100:.2f}%")
    
    # Plot results
    # Plot 1: Epsilon comparison
    ax1.plot(results['sample_sizes'], results['sampled_epsilons'], 'b-', label='Sampled ε')
    ax1.plot(results['sample_sizes'], results['theoretical_epsilons'], 'r--', label='Theoretical ε')
    ax1.set_xscale('log')
    ax1.set_xlabel('Sample Size (N)')
    ax1.set_ylabel('Optimal Epsilon (ε)')
    ax1.set_title(f'{name} Distribution: Epsilon Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Welfare comparison
    ax2.plot(results['sample_sizes'], results['sampled_welfares'], 'b-', label='Sampled Welfare')
    ax2.plot(results['sample_sizes'], results['theoretical_welfares'], 'r--', label='Theoretical Welfare')
    ax2.set_xscale('log')
    ax2.set_xlabel('Sample Size (N)')
    ax2.set_ylabel('Welfare')
    ax2.set_title(f'{name} Distribution: Welfare Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    # Define sample sizes to test using linspace but convert to integers
    sample_sizes = np.linspace(100, 1000000, 20)  # 20 points between 100 and 1,000,000
    sample_sizes = np.unique(np.round(sample_sizes).astype(int))  # Convert to unique integers
    strict_penalty_factor = 1.0
    
    # Test Uniform distribution
    print("\n=== Testing Uniform Distribution ===")
    uniform_params = {'low': 0.00000001, 'high': 1.0}
    uniform_results = run_simulation_comparison(
        uniform_preferences, "Uniform", uniform_params, sample_sizes, strict_penalty_factor
    )
    
    # Test Normal distribution
    print("\n=== Testing Normal Distribution ===")
    normal_params = {'mu': 0.5, 'sigma': 0.1}
    normal_results = run_simulation_comparison(
        normal_preferences, "Normal", normal_params, sample_sizes, strict_penalty_factor
    )
    
    # Test Mixture of Normals
    print("\n=== Testing Mixture of Normals ===")
    mixture_params = {
        'mus': [0.2, 0.8],
        'sigmas': [0.05, 0.05],
        'weights': [0.5, 0.5]
    }
    mixture_results = run_simulation_comparison(
        mixture_normal_preferences, "Mixture of Normals", mixture_params, sample_sizes, strict_penalty_factor
    )
    
    # Print summary of convergence
    print("\n=== Summary of Convergence ===")
    for name, results in [("Uniform", uniform_results), ("Normal", normal_results), ("Mixture", mixture_results)]:
        print(f"\n{name} Distribution:")
        for i in range(len(sample_sizes)-1):
            N1, N2 = sample_sizes[i], sample_sizes[i+1]
            eps_diff1 = abs(results['sampled_epsilons'][i] - results['theoretical_epsilons'][i])
            eps_diff2 = abs(results['sampled_epsilons'][i+1] - results['theoretical_epsilons'][i+1])
            welfare_diff1 = abs(results['sampled_welfares'][i] - results['theoretical_welfares'][i])
            welfare_diff2 = abs(results['sampled_welfares'][i+1] - results['theoretical_welfares'][i+1])
            
            print(f"N={N1} to N={N2}:")
            print(f"  Epsilon difference reduction: {eps_diff1/eps_diff2:.2f}x")
            print(f"  Welfare difference reduction: {welfare_diff1/welfare_diff2:.2f}x")

if __name__ == "__main__":
    main() 
