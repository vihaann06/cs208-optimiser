import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import matplotlib.pyplot as plt

def uniform_preferences(n_samples, low=0.00001, high=4.0):
    """Generate uniformly distributed preferences"""
    return np.random.uniform(low=low, high=high, size=n_samples)

def normal_preferences(n_samples, mu=0.5, sigma=0.15, low=0.00001, high=4.0):
    """
    Generate normally distributed preferences
    Truncated to [low,high] range since preferences should be bounded
    """
    samples = np.random.normal(mu, sigma, size=n_samples)
    return np.clip(samples, low, high)

def mixture_normal_preferences(n_samples, mus=[0.3, 0.7], sigmas=[0.1, 0.1], 
                             weights=[0.4, 0.6], low=0.00001, high=4.0):
    """
    Generate preferences from a mixture of normal distributions
    """
    n_components = len(mus)
    component_indices = np.random.choice(n_components, size=n_samples, p=weights)
    samples = np.zeros(n_samples)
    
    for i in range(n_components):
        mask = (component_indices == i)
        n_comp_samples = mask.sum()
        samples[mask] = np.random.normal(mus[i], sigmas[i], size=n_comp_samples)
    
    return np.clip(samples, low, high)

def calculate_W(epsilon, epsilon_a, epsilon_b, N, strict_penalty_factor=1.0):
    """
    Calculate the welfare function W for given parameters.
    
    Args:
        epsilon (float): The privacy parameter to evaluate
        epsilon_a (float): Lower bound of epsilon distribution
        epsilon_b (float): Upper bound of epsilon distribution
        N (int): Total number of potential participants
        strict_penalty_factor (float): Factor to control penalty for strict privacy
                                     Higher values mean stronger penalty for small epsilon
    
    Returns:
        float: The value of the welfare function W
    """
    
    # Ensure epsilon is at least epsilon_a since the bounds in the integral are max(epsilon, epsilon_a)
    lower_bound = max(epsilon, epsilon_a)
    
    # Define the inner function for the second part of the first term
    def inner_integral(y):
        return 1/y
    
    # Calculate the inner integral once (doesn't depend on x)
    inner_result = integrate.quad(inner_integral, lower_bound, epsilon_b)[0]
    
    # Define the function for the first term's outer integral with strict penalty
    def first_term_integrand(x):
        # Apply penalty to first term as well
        base_utility = (1/x) * (1 - 1/(N * epsilon))
        
        # Apply a much stronger penalty for small epsilon values
        # This creates a dramatic drop in utility when epsilon is too small
        if epsilon < epsilon_a * 2:  # Threshold for "too strict" privacy
            penalty = np.exp(-strict_penalty_factor * (epsilon_a/epsilon))
        else:
            penalty = 1.0
            
        return base_utility * penalty * inner_result
    
    # Define the function for the second term's integral with strict penalty
    def second_term_integrand(x):
        # Base utility
        base_utility = 1 - epsilon/x
        
        # Apply a much stronger penalty for small epsilon values
        if epsilon < epsilon_a * 2:  # Threshold for "too strict" privacy
            # This creates a dramatic drop in utility when epsilon is too small
            penalty = np.exp(-strict_penalty_factor * (epsilon_a/epsilon))
            
            # Additional penalty based on the ratio of user preference to epsilon
            # This creates a stronger penalty when epsilon is much smaller than user preference
            ratio_penalty = np.exp(-strict_penalty_factor * (x/epsilon))
            
            return base_utility * penalty * ratio_penalty
        else:
            return base_utility
    
    # Calculate the first term (double integral)
    first_term = integrate.quad(first_term_integrand, epsilon_a, epsilon_b)[0]
    
    # Calculate the second term (single integral)
    second_term = integrate.quad(second_term_integrand, lower_bound, epsilon_b)[0]
    
    return first_term + second_term

def find_optimal_epsilon(epsilon_a, epsilon_b, M, strict_penalty_factor=1.0):
    """
    Find the value of epsilon that maximizes W.
    
    Args:
        epsilon_a (float): Lower bound of epsilon distribution
        epsilon_b (float): Upper bound of epsilon distribution
        M (int): Total number of potential participants
        strict_penalty_factor (float): Factor to control penalty for strict privacy
    
    Returns:
        tuple: (optimal_epsilon, maximum_W)
    """
    def objective(epsilon):
        # Handle the case where epsilon is below epsilon_a
        if epsilon < epsilon_a:
            return float('inf')  # Invalid region
        return -calculate_W(epsilon, epsilon_a, epsilon_b, M, strict_penalty_factor)
    
    # Find the minimum of the negative function (equivalent to finding maximum of W)
    result = minimize_scalar(objective, 
                           bounds=(epsilon_a, epsilon_b),
                           method='bounded')
    
    return result.x, -result.fun  # Return the optimal epsilon and the maximum W value

def analyze_distribution(distribution_func, name, params=None, strict_penalty_factor=1.0):
    """
    Analyze a preference distribution and find optimal epsilon
    """
    if params is None:
        params = {}
    
    # Generate samples
    samples = distribution_func(**params)
    epsilon_a = np.min(samples)
    epsilon_b = np.max(samples)
    N = len(samples)
    
    # Find optimal epsilon
    optimal_epsilon, max_welfare = find_optimal_epsilon(epsilon_a, epsilon_b, N, strict_penalty_factor)
    
    print(f"\nResults for {name} distribution (strict_penalty_factor={strict_penalty_factor}):")
    print(f"Optimal epsilon: {optimal_epsilon:.4f}")
    print(f"Maximum welfare: {max_welfare:.4f}")
    
    # Plot distribution
    plt.figure(figsize=(8, 4))
    plt.hist(samples, bins=30, density=True, alpha=0.7)
    plt.axvline(optimal_epsilon, color='r', linestyle='--', label=f'Optimal ε: {optimal_epsilon:.4f}')
    plt.title(f'{name} Distribution of Privacy Preferences\n(strict_penalty_factor={strict_penalty_factor})')
    plt.xlabel('Privacy Preference (ε)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Plot welfare function for a range of epsilon values
    epsilon_range = np.linspace(epsilon_a, epsilon_b, 100)
    welfare_values = [calculate_W(eps, epsilon_a, epsilon_b, N, strict_penalty_factor) for eps in epsilon_range]
    
    plt.figure(figsize=(8, 4))
    plt.plot(epsilon_range, welfare_values)
    plt.axvline(optimal_epsilon, color='r', linestyle='--', label=f'Optimal ε: {optimal_epsilon:.4f}')
    plt.title(f'Welfare Function for {name} Distribution\n(strict_penalty_factor={strict_penalty_factor})')
    plt.xlabel('Privacy Parameter (ε)')
    plt.ylabel('Welfare')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Parameters for different distributions
    n_samples = 1000
    
    # Test different strict penalty factors with wider range
    penalty_factors = [1.0, 5.0, 10.0, 20.0, 50.0]
    
    for penalty in penalty_factors:
        print(f"\n=== Testing with strict_penalty_factor = {penalty} ===")
        
        # Uniform distribution
        analyze_distribution(uniform_preferences, "Uniform", 
                            {'n_samples': n_samples, 'low': 0.0000001, 'high': 4.0},
                            strict_penalty_factor=penalty)
        
        # Normal distribution
        analyze_distribution(normal_preferences, "Normal",
                            {'n_samples': n_samples, 'mu': 0.01, 'sigma': 0.4},
                            strict_penalty_factor=penalty)
        
        # Mixture of normals
        analyze_distribution(mixture_normal_preferences, "Mixture of Normals",
                            {'n_samples': n_samples, 
                             'mus': [0.01, 0.5],
                             'sigmas': [0.5, 0.2],
                             'weights': [0.6, 0.4]},
                            strict_penalty_factor=penalty)
