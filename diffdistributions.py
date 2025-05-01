import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import matplotlib.pyplot as plt

def uniform_preferences(n_samples, low=0.00000001, high=1.0):
    """Generate uniformly distributed preferences"""
    return np.random.uniform(low=low, high=high, size=n_samples)

def normal_preferences(n_samples, mu=0.5, sigma=0.15, low=0.00000001, high=1.0):
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
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot histogram
    ax1.hist(samples, bins=30, alpha=0.7)
    ax1.axvline(optimal_epsilon, color='r', linestyle='--', label=f'Optimal ε: {optimal_epsilon:.4f}')
    ax1.set_title(f'Sampled Distribution\nN={N}')
    ax1.set_xlabel('Privacy Preference (ε)')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # Plot theoretical PDF
    x = np.linspace(epsilon_a, epsilon_b, 1000)
    if name == "Uniform":
        pdf = np.ones_like(x) / (epsilon_b - epsilon_a)
    elif name == "Normal":
        pdf = norm.pdf(x, loc=params['mu'], scale=params['sigma'])
        # Normalize to match the histogram scale
        pdf = pdf * N * (epsilon_b - epsilon_a) / len(x)
    elif name == "Mixture of Normals":
        pdf = np.zeros_like(x)
        for mu, sigma, weight in zip(params['mus'], params['sigmas'], params['weights']):
            pdf += weight * norm.pdf(x, loc=mu, scale=sigma)
        # Normalize to match the histogram scale
        pdf = pdf * N * (epsilon_b - epsilon_a) / len(x)
    
    ax2.plot(x, pdf, 'r-', linewidth=2)
    ax2.axvline(optimal_epsilon, color='r', linestyle='--', label=f'Optimal ε: {optimal_epsilon:.4f}')
    ax2.set_title('Theoretical PDF')
    ax2.set_xlabel('Privacy Preference (ε)')
    ax2.set_ylabel('Density')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot welfare function for a range of epsilon values
    epsilon_range = np.linspace(epsilon_a, epsilon_b, 100)
    welfare_values = [calculate_W(eps, epsilon_a, epsilon_b, N, strict_penalty_factor) for eps in epsilon_range]
    
    plt.figure(figsize=(8, 4))
    plt.plot(epsilon_range, welfare_values)
    plt.axvline(optimal_epsilon, color='r', linestyle='--', label=f'Optimal ε: {optimal_epsilon:.4f}')
    plt.title(f'Welfare Function for {name} Distribution\nN={N}, strict_penalty_factor={strict_penalty_factor}')
    plt.xlabel('Privacy Parameter (ε)')
    plt.ylabel('Welfare')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the range of welfare values for context
    print(f"Welfare range: [{min(welfare_values):.2f}, {max(welfare_values):.2f}]")

def calculate_theoretical_epsilon(name, params, N, strict_penalty_factor=1.0):
    """
    Calculate the theoretical optimal epsilon for a given distribution using the welfare function
    """
    # Define theoretical bounds based on distribution type
    if name == "Uniform":
        epsilon_a = params['low']
        epsilon_b = params['high']
        def theoretical_pdf(x):
            return 1 / (epsilon_b - epsilon_a)
    elif name == "Normal":
        epsilon_a = 0.00000001  # Same as in normal_preferences
        epsilon_b = 1.0  # Same as in normal_preferences
        def theoretical_pdf(x):
            return norm.pdf(x, loc=params['mu'], scale=params['sigma'])
    elif name == "Mixture of Normals":
        # Use bounds that cover 99.7% of the distribution (3 standard deviations from each mean)
        epsilon_a = min(params['mus']) - 3 * max(params['sigmas'])
        epsilon_b = max(params['mus']) + 3 * max(params['sigmas'])
        # Ensure bounds are positive and reasonable
        epsilon_a = max(0.00001, epsilon_a)
        epsilon_b = min(1.0, epsilon_b)
        
        # Define the theoretical PDF for mixture of normals
        def theoretical_pdf(x):
            pdf = 0
            for mu, sigma, weight in zip(params['mus'], params['sigmas'], params['weights']):
                # Calculate each normal component using the exact formula
                component = weight * (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
                pdf += component
            return pdf
    
    def theoretical_W(epsilon):
        # Calculate the welfare function using the theoretical PDF
        lower_bound = max(epsilon, epsilon_a)
        
        # First term
        def first_term_integrand(x):
            base_utility = (1/x) * (1 - 1/(N * epsilon))
            if epsilon < epsilon_a * 2:
                penalty = np.exp(-strict_penalty_factor * (epsilon_a/epsilon))
            else:
                penalty = 1.0
            return base_utility * penalty * theoretical_pdf(x)
        
        # Second term
        def second_term_integrand(x):
            base_utility = 1 - epsilon/x
            if epsilon < epsilon_a * 2:
                penalty = np.exp(-strict_penalty_factor * (epsilon_a/epsilon))
                ratio_penalty = np.exp(-strict_penalty_factor * (x/epsilon))
                return base_utility * penalty * ratio_penalty * theoretical_pdf(x)
            else:
                return base_utility * theoretical_pdf(x)
        
        first_term = integrate.quad(first_term_integrand, epsilon_a, epsilon_b)[0]
        second_term = integrate.quad(second_term_integrand, lower_bound, epsilon_b)[0]
        return first_term + second_term
    
    # Find the epsilon that maximizes the theoretical welfare
    def objective(epsilon):
        if epsilon < epsilon_a:
            return float('inf')
        return -theoretical_W(epsilon)
    
    # Use a more robust optimization method with multiple starting points
    epsilons = np.linspace(epsilon_a, epsilon_b, 10)
    best_epsilon = None
    best_welfare = float('-inf')
    
    for start_epsilon in epsilons:
        result = minimize_scalar(objective, 
                               bounds=(epsilon_a, epsilon_b),
                               method='bounded',
                               options={'xatol': 1e-8})
        if -result.fun > best_welfare:
            best_welfare = -result.fun
            best_epsilon = result.x
    
    return best_epsilon

def compare_distributions(distribution_func, name, params=None, strict_penalty_factor=1.0, n_trials=10):
    """
    Compare the same distribution for N=1000 and N=10000 over multiple trials
    """
    if params is None:
        params = {}
    
    # Create two separate figures
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Store results for averaging
    n1000_results = {'sampled_epsilons': [], 'sampled_welfares': [], 'theoretical_epsilons': []}
    n10000_results = {'sampled_epsilons': [], 'sampled_welfares': [], 'theoretical_epsilons': []}
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Test for N=1000
        n_samples = 1000
        samples = distribution_func(**{**params, 'n_samples': n_samples})
        epsilon_a = np.min(samples)
        epsilon_b = np.max(samples)
        sampled_optimal_epsilon, sampled_welfare = find_optimal_epsilon(epsilon_a, epsilon_b, n_samples, strict_penalty_factor)
        theoretical_optimal_epsilon = calculate_theoretical_epsilon(name, params, n_samples, strict_penalty_factor)
        
        # Store results
        n1000_results['sampled_epsilons'].append(sampled_optimal_epsilon)
        n1000_results['sampled_welfares'].append(sampled_welfare)
        n1000_results['theoretical_epsilons'].append(theoretical_optimal_epsilon)
        
        # Calculate welfare values for plotting
        epsilon_range = np.linspace(epsilon_a, epsilon_b, 100)
        sampled_welfare_values = [calculate_W(eps, epsilon_a, epsilon_b, n_samples, strict_penalty_factor) for eps in epsilon_range]
        
        # Plot N=1000 histogram (only for first trial)
        if trial == 0:
            ax1.hist(samples, bins=30, alpha=0.7)
            ax1.axvline(sampled_optimal_epsilon, color='r', linestyle='--', label=f'Sampled ε: {sampled_optimal_epsilon:.4f}')
            ax1.axvline(theoretical_optimal_epsilon, color='g', linestyle='--', label=f'Theoretical ε: {theoretical_optimal_epsilon:.4f}')
            ax1.set_title(f'Sampled Distribution\nN=1000')
            ax1.set_xlabel('Privacy Preference (ε)')
            ax1.set_ylabel('Count')
            ax1.legend()
            
            # Plot N=1000 theoretical PDF
            x = np.linspace(epsilon_a, epsilon_b, 1000)
            if name == "Uniform":
                pdf = np.ones_like(x) / (epsilon_b - epsilon_a)
            elif name == "Normal":
                pdf = norm.pdf(x, loc=params['mu'], scale=params['sigma'])
                pdf = pdf * n_samples * (epsilon_b - epsilon_a) / len(x)
            elif name == "Mixture of Normals":
                pdf = np.zeros_like(x)
                total_weight = sum(params['weights'])
                for mu, sigma, weight in zip(params['mus'], params['sigmas'], params['weights']):
                    normalized_weight = weight / total_weight
                    component_pdf = normalized_weight * norm.pdf(x, loc=mu, scale=sigma)
                    pdf += component_pdf
                    ax2.plot(x, component_pdf * n_samples * (epsilon_b - epsilon_a) / len(x), 
                            '--', alpha=0.5, label=f'Component μ={mu:.2f}, σ={sigma:.2f}')
                ax2.plot(x, pdf * n_samples * (epsilon_b - epsilon_a) / len(x), 'r-', linewidth=2, label='Total Mixture')
            else:
                pdf = np.zeros_like(x)
            
            if name != "Mixture of Normals":
                ax2.plot(x, pdf, 'r-', linewidth=2)
            
            ax2.axvline(sampled_optimal_epsilon, color='r', linestyle='--', label=f'Sampled ε: {sampled_optimal_epsilon:.4f}')
            ax2.axvline(theoretical_optimal_epsilon, color='g', linestyle='--', label=f'Theoretical ε: {theoretical_optimal_epsilon:.4f}')
            ax2.set_title('Theoretical PDF\nN=1000')
            ax2.set_xlabel('Privacy Preference (ε)')
            ax2.set_ylabel('Density')
            ax2.legend()
        
        # Plot welfare function for N=1000
        ax5.plot(epsilon_range, sampled_welfare_values, 'b-', alpha=0.3)
        ax5.axvline(sampled_optimal_epsilon, color='r', linestyle='--', alpha=0.3)
        ax5.axvline(theoretical_optimal_epsilon, color='g', linestyle='--', alpha=0.3)
        
        # Test for N=10000
        n_samples = 10000
        samples = distribution_func(**{**params, 'n_samples': n_samples})
        epsilon_a = np.min(samples)
        epsilon_b = np.max(samples)
        sampled_optimal_epsilon, sampled_welfare = find_optimal_epsilon(epsilon_a, epsilon_b, n_samples, strict_penalty_factor)
        theoretical_optimal_epsilon = calculate_theoretical_epsilon(name, params, n_samples, strict_penalty_factor)
        
        # Store results
        n10000_results['sampled_epsilons'].append(sampled_optimal_epsilon)
        n10000_results['sampled_welfares'].append(sampled_welfare)
        n10000_results['theoretical_epsilons'].append(theoretical_optimal_epsilon)
        
        # Calculate welfare values for plotting
        epsilon_range = np.linspace(epsilon_a, epsilon_b, 100)
        sampled_welfare_values = [calculate_W(eps, epsilon_a, epsilon_b, n_samples, strict_penalty_factor) for eps in epsilon_range]
        
        # Plot N=10000 histogram (only for first trial)
        if trial == 0:
            ax3.hist(samples, bins=30, alpha=0.7)
            ax3.axvline(sampled_optimal_epsilon, color='r', linestyle='--', label=f'Sampled ε: {sampled_optimal_epsilon:.4f}')
            ax3.axvline(theoretical_optimal_epsilon, color='g', linestyle='--', label=f'Theoretical ε: {theoretical_optimal_epsilon:.4f}')
            ax3.set_title(f'Sampled Distribution\nN=10000')
            ax3.set_xlabel('Privacy Preference (ε)')
            ax3.set_ylabel('Count')
            ax3.legend()
            
            # Plot N=10000 theoretical PDF
            x = np.linspace(epsilon_a, epsilon_b, 1000)
            if name == "Uniform":
                pdf = np.ones_like(x) / (epsilon_b - epsilon_a)
            elif name == "Normal":
                pdf = norm.pdf(x, loc=params['mu'], scale=params['sigma'])
                pdf = pdf * n_samples * (epsilon_b - epsilon_a) / len(x)
            elif name == "Mixture of Normals":
                pdf = np.zeros_like(x)
                total_weight = sum(params['weights'])
                for mu, sigma, weight in zip(params['mus'], params['sigmas'], params['weights']):
                    normalized_weight = weight / total_weight
                    component_pdf = normalized_weight * norm.pdf(x, loc=mu, scale=sigma)
                    pdf += component_pdf
                    ax4.plot(x, component_pdf * n_samples * (epsilon_b - epsilon_a) / len(x), 
                            '--', alpha=0.5, label=f'Component μ={mu:.2f}, σ={sigma:.2f}')
                ax4.plot(x, pdf * n_samples * (epsilon_b - epsilon_a) / len(x), 'r-', linewidth=2, label='Total Mixture')
            else:
                pdf = np.zeros_like(x)
            
            if name != "Mixture of Normals":
                ax4.plot(x, pdf, 'r-', linewidth=2)
            
            ax4.axvline(sampled_optimal_epsilon, color='r', linestyle='--', label=f'Sampled ε: {sampled_optimal_epsilon:.4f}')
            ax4.axvline(theoretical_optimal_epsilon, color='g', linestyle='--', label=f'Theoretical ε: {theoretical_optimal_epsilon:.4f}')
            ax4.set_title('Theoretical PDF\nN=10000')
            ax4.set_xlabel('Privacy Preference (ε)')
            ax4.set_ylabel('Density')
            ax4.legend()
        
        # Plot welfare function for N=10000
        ax6.plot(epsilon_range, sampled_welfare_values, 'b-', alpha=0.3)
        ax6.axvline(sampled_optimal_epsilon, color='r', linestyle='--', alpha=0.3)
        ax6.axvline(theoretical_optimal_epsilon, color='g', linestyle='--', alpha=0.3)
    
    # Add final touches to welfare plots
    ax5.set_title('Welfare Functions\nN=1000 (10 trials)')
    ax5.set_xlabel('Privacy Parameter (ε)')
    ax5.set_ylabel('Welfare')
    ax5.legend(['Sampled Welfare', 'Sampled ε', 'Theoretical ε'])
    
    ax6.set_title('Welfare Functions\nN=10000 (10 trials)')
    ax6.set_xlabel('Privacy Parameter (ε)')
    ax6.set_ylabel('Welfare')
    ax6.legend(['Sampled Welfare', 'Sampled ε', 'Theoretical ε'])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary statistics for {name} distribution over {n_trials} trials:")
    print(f"N=1000:")
    print(f"  Sampled ε: mean = {np.mean(n1000_results['sampled_epsilons']):.4f}, std = {np.std(n1000_results['sampled_epsilons']):.4f}")
    print(f"  Sampled Welfare: mean = {np.mean(n1000_results['sampled_welfares']):.4f}, std = {np.std(n1000_results['sampled_welfares']):.4f}")
    print(f"  Theoretical ε: mean = {np.mean(n1000_results['theoretical_epsilons']):.4f}, std = {np.std(n1000_results['theoretical_epsilons']):.4f}")
    print(f"N=10000:")
    print(f"  Sampled ε: mean = {np.mean(n10000_results['sampled_epsilons']):.4f}, std = {np.std(n10000_results['sampled_epsilons']):.4f}")
    print(f"  Sampled Welfare: mean = {np.mean(n10000_results['sampled_welfares']):.4f}, std = {np.std(n10000_results['sampled_welfares']):.4f}")
    print(f"  Theoretical ε: mean = {np.mean(n10000_results['theoretical_epsilons']):.4f}, std = {np.std(n10000_results['theoretical_epsilons']):.4f}")

# Example usage
if __name__ == "__main__":
    # Test different strict penalty factors with wider range
    penalty_factors = [1.0]
    
    for penalty in penalty_factors:
        print(f"\n=== Testing with strict_penalty_factor = {penalty} ===")
        
        # Compare distributions for N=1000 and N=10000
        print("\n=== Uniform Distribution ===")
        compare_distributions(uniform_preferences, "Uniform", 
                            {'low':0.00000001, 'high': 1.0},
                            strict_penalty_factor=penalty)
        
        print("\n=== Normal Distribution ===")
        compare_distributions(normal_preferences, "Normal",
                            {'mu': 0.5, 'sigma': 0.1},
                            strict_penalty_factor=penalty)
        
        print("\n=== Mixture of Normals ===")
        compare_distributions(mixture_normal_preferences, "Mixture of Normals",
                            {'mus': [0.4, 0.6],  # More separated means
                             'sigmas': [0.1, 0.1],  # Smaller standard deviations
                             'weights': [0.5, 0.5]},  # Equal weights
                            strict_penalty_factor=penalty)
