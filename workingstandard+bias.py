import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon

def calculate_W_equation1(N, epsilon_a, epsilon_b, n, epsilon_values, pdf_func):
    """
    Calculate W using the first equation:
    W = N ∫(ε_a to ε_b) (1/x)(1-1/(nε)) ∫(max(ε,ε_a) to ε_b) (1/y) dy dx 
        + ∫(max(ε,ε_a) to ε_b) (1-ε/x) dx
    
    Parameters:
    - N: coefficient
    - epsilon_a: lower bound
    - epsilon_b: upper bound
    - n: parameter in the equation
    - epsilon_values: array of epsilon values to evaluate W for
    - pdf_func: function to calculate the PDF of the distribution
    
    Returns:
    - Array of W values corresponding to each epsilon
    """
    results = []
    
    for epsilon in epsilon_values:
        # Define the inner integral function
        def inner_integral(x):
            max_val = max(epsilon, epsilon_a)
            if max_val >= epsilon_b:
                return 0  # No contribution if max_val >= epsilon_b
            
            # Inner integral: ∫(max(ε,ε_a) to ε_b) (1/y) dy = ln(ε_b) - ln(max(ε,ε_a))
            inner_result = np.log(epsilon_b) - np.log(max_val)
            return inner_result
        
        # Define the first part of the outer integral
        def first_part_integrand(x):
            if x < epsilon_a or x > epsilon_b:
                return 0
            
            factor = (1/x) * (1 - 1/(n*epsilon))
            return factor * inner_integral(x) * pdf_func(x)
        
        # Define the second part of the integral
        def second_part_integrand(x):
            if x < max(epsilon, epsilon_a) or x > epsilon_b:
                return 0
            return (1 - epsilon/x) * pdf_func(x)
        
        # Compute the integrals
        first_part, _ = integrate.quad(first_part_integrand, epsilon_a, epsilon_b)
        
        max_val = max(epsilon, epsilon_a)
        if max_val < epsilon_b:
            second_part, _ = integrate.quad(second_part_integrand, max_val, epsilon_b)
        else:
            second_part = 0
        
        # Combine results
        W = N * first_part + second_part
        results.append(W)
    
    return np.array(results)

def calculate_omega(epsilon, epsilon_a, epsilon_b):
    """
    Calculate Ω = [∫(ε_a to ε_b) (1/z) dz/(ε_b - ε_a) - ∫(ε to ε_b) (1/z) dz/(ε_b - ε)]
    """
    # First integral: ∫(ε_a to ε_b) (1/z) dz/(ε_b - ε_a)
    first_integral = (np.log(epsilon_b) - np.log(epsilon_a)) / (epsilon_b - epsilon_a)
    
    # Second integral: ∫(ε to ε_b) (1/z) dz/(ε_b - ε)
    if epsilon < epsilon_b:
        second_integral = (np.log(epsilon_b) - np.log(epsilon)) / (epsilon_b - epsilon)
    else:
        second_integral = 0
    
    return first_integral - second_integral

def calculate_W_equation2(N, epsilon_a, epsilon_b, n, c, epsilon_values, pdf_func):
    """
    Calculate W using the second equation:
    W = N ∫(ε_a to ε_b) ((1/x)(1-1/(nε)) + c*Ω) ∫(max(ε,ε_a) to ε_b) (1/y) dy dx 
        + ∫(max(ε,ε_a) to ε_b) (1-ε/x) dx
    
    Parameters:
    - N: coefficient
    - epsilon_a: lower bound
    - epsilon_b: upper bound
    - n: parameter in the equation
    - c: coefficient for Ω
    - epsilon_values: array of epsilon values to evaluate W for
    - pdf_func: function to calculate the PDF of the distribution
    
    Returns:
    - Array of W values corresponding to each epsilon
    """
    results = []
    
    for epsilon in epsilon_values:
        omega = calculate_omega(epsilon, epsilon_a, epsilon_b)
        
        # Define the inner integral function
        def inner_integral(x):
            max_val = max(epsilon, epsilon_a)
            if max_val >= epsilon_b:
                return 0
            
            # Inner integral: ∫(max(ε,ε_a) to ε_b) (1/y) dy = ln(ε_b) - ln(max(ε,ε_a))
            inner_result = np.log(epsilon_b) - np.log(max_val)
            return inner_result
        
        # Define the first part of the outer integral
        def first_part_integrand(x):
            if x < epsilon_a or x > epsilon_b:
                return 0
            
            factor = (1/x) * (1 - 1/(n*epsilon)) + c * omega
            return factor * inner_integral(x) * pdf_func(x)
        
        # Define the second part of the integral
        def second_part_integrand(x):
            if x < max(epsilon, epsilon_a) or x > epsilon_b:
                return 0
            return (1 - epsilon/x) * pdf_func(x)
        
        # Compute the integrals
        first_part, _ = integrate.quad(first_part_integrand, epsilon_a, epsilon_b)
        
        max_val = max(epsilon, epsilon_a)
        if max_val < epsilon_b:
            second_part, _ = integrate.quad(second_part_integrand, max_val, epsilon_b)
        else:
            second_part = 0
        
        # Combine results
        W = N * first_part + second_part
        results.append(W)
    
    return np.array(results)

def find_optimal_epsilon(N, epsilon_a, epsilon_b, n, c, equation_type, pdf_func):
    """
    Find the optimal epsilon that maximizes W for a given equation type
    """
    def objective(epsilon):
        if epsilon < epsilon_a:
            return float('inf')
        if equation_type == 1:
            W = calculate_W_equation1(N, epsilon_a, epsilon_b, n, [epsilon], pdf_func)[0]
        else:
            W = calculate_W_equation2(N, epsilon_a, epsilon_b, n, c, [epsilon], pdf_func)[0]
        return -W  # Negative because we want to maximize W

    result = minimize_scalar(objective, 
                            bounds=(epsilon_a, epsilon_b),
                            method='bounded',
                            options={'xatol': 1e-8})
    return result.x, -result.fun

def plot_theoretical_distribution(dist_type, params, epsilon_a, epsilon_b, title, custom_pdf=None):
    """
    Plot the theoretical probability density function for a given distribution
    """
    x = np.linspace(epsilon_a, epsilon_b, 1000)
    if custom_pdf is not None:
        pdf = custom_pdf(x)
    elif dist_type == 'uniform':
        pdf = uniform.pdf(x, loc=epsilon_a, scale=epsilon_b-epsilon_a)
    elif dist_type == 'normal':
        pdf = norm.pdf(x, loc=params['mu'], scale=params['sigma'])
    elif dist_type == 'exponential':
        pdf = expon.pdf(x-epsilon_a, scale=params['scale'])
    else:
        raise ValueError("Unknown distribution type")
    plt.figure(figsize=(12, 4))
    plt.plot(x, pdf, 'b-', linewidth=2)
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Probability Density')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_single_W(epsilon_values, W_values, opt_epsilon, title, color='b'):
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_values, W_values, color+'-', label=title)
    plt.axvline(opt_epsilon, color=color, linestyle=':', label=f'Optimal ε: {opt_epsilon:.8f}')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('W')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def reversed_expon_pdf(x, a, b, scale):
    # Normalization constant so that the integral over [a, b] is 1
    norm = 1 - np.exp(-(b - a) / scale)
    return np.where((x >= a) & (x <= b), np.exp(-(b - x) / scale) / (scale * norm), 0)

# Example usage
if __name__ == "__main__":
    # Define parameters
    N = 1.0        # Normalization factor
    epsilon_a = 0.00000001  # Lower bound
    epsilon_b = 1.0  # Upper bound
    n = 5.0        # Parameter in equation
    c = 0.1        # Coefficient for Omega term
    num_points = 100  # Number of points for evaluation
    
    # Create evaluation points
    epsilon_values = np.linspace(epsilon_a, epsilon_b, num_points)
    
    # 1. Uniform Distribution
    print("\n=== Uniform Distribution ===")
    plot_theoretical_distribution('uniform', {}, epsilon_a, epsilon_b, 
                                "Theoretical Uniform Distribution")
    
    # Calculate W for uniform distribution
    W_uniform1 = calculate_W_equation1(N, epsilon_a, epsilon_b, n, epsilon_values, lambda x: uniform.pdf(x, loc=epsilon_a, scale=epsilon_b-epsilon_a))
    W_uniform2 = calculate_W_equation2(N, epsilon_a, epsilon_b, n, c, epsilon_values, lambda x: uniform.pdf(x, loc=epsilon_a, scale=epsilon_b-epsilon_a))
    
    # Find optimal epsilons for uniform distribution
    opt_epsilon_uniform1, max_welfare_uniform1 = find_optimal_epsilon(N, epsilon_a, epsilon_b, n, c, 1, lambda x: uniform.pdf(x, loc=epsilon_a, scale=epsilon_b-epsilon_a))
    opt_epsilon_uniform2, max_welfare_uniform2 = find_optimal_epsilon(N, epsilon_a, epsilon_b, n, c, 2, lambda x: uniform.pdf(x, loc=epsilon_a, scale=epsilon_b-epsilon_a))
    
    # Print results for uniform distribution
    print("\nOptimal Epsilon Values (Uniform Distribution):")
    print(f"Equation 1: ε = {opt_epsilon_uniform1:.8f}, W = {max_welfare_uniform1:.8f}")
    print(f"Equation 2: ε = {opt_epsilon_uniform2:.8f}, W = {max_welfare_uniform2:.8f}")
    print(f"Difference: {abs(opt_epsilon_uniform1 - opt_epsilon_uniform2):.8f}")
    
    # Plot uniform distribution results
    plot_single_W(epsilon_values, W_uniform1, opt_epsilon_uniform1, "W1 for Uniform Distribution", color='b')
    plot_single_W(epsilon_values, W_uniform2, opt_epsilon_uniform2, "W2 for Uniform Distribution", color='r')
    
    # 2. Normal Distribution
    print("\n=== Normal Distribution ===")
    normal_params = {'mu': 0.5, 'sigma': 0.3}
    plot_theoretical_distribution('normal', normal_params, epsilon_a, epsilon_b,
                                "Theoretical Normal Distribution")
    
    # Calculate W for normal distribution
    W_normal1 = calculate_W_equation1(N, epsilon_a, epsilon_b, n, epsilon_values, lambda x: norm.pdf(x, loc=normal_params['mu'], scale=normal_params['sigma']))
    W_normal2 = calculate_W_equation2(N, epsilon_a, epsilon_b, n, c, epsilon_values, lambda x: norm.pdf(x, loc=normal_params['mu'], scale=normal_params['sigma']))
    
    # Find optimal epsilons for normal distribution
    opt_epsilon_normal1, max_welfare_normal1 = find_optimal_epsilon(N, epsilon_a, epsilon_b, n, c, 1, lambda x: norm.pdf(x, loc=normal_params['mu'], scale=normal_params['sigma']))
    opt_epsilon_normal2, max_welfare_normal2 = find_optimal_epsilon(N, epsilon_a, epsilon_b, n, c, 2, lambda x: norm.pdf(x, loc=normal_params['mu'], scale=normal_params['sigma']))
    
    # Print results for normal distribution
    print("\nOptimal Epsilon Values (Normal Distribution):")
    print(f"Equation 1: ε = {opt_epsilon_normal1:.8f}, W = {max_welfare_normal1:.8f}")
    print(f"Equation 2: ε = {opt_epsilon_normal2:.8f}, W = {max_welfare_normal2:.8f}")
    print(f"Difference: {abs(opt_epsilon_normal1 - opt_epsilon_normal2):.8f}")
    
    # Plot normal distribution results
    plot_single_W(epsilon_values, W_normal1, opt_epsilon_normal1, "W1 for Normal Distribution", color='b')
    plot_single_W(epsilon_values, W_normal2, opt_epsilon_normal2, "W2 for Normal Distribution", color='r')
    
    # 3. Exponential Distribution
    print("\n=== Exponential Distribution ===")
    exp_params = {'scale': 0.5}
    plot_theoretical_distribution('exponential', exp_params, epsilon_a, epsilon_b,
                                "Theoretical Exponential Distribution")
    
    # Calculate W for exponential distribution
    W_exp1 = calculate_W_equation1(N, epsilon_a, epsilon_b, n, epsilon_values, lambda x: expon.pdf(x-epsilon_a, scale=exp_params['scale']))
    W_exp2 = calculate_W_equation2(N, epsilon_a, epsilon_b, n, c, epsilon_values, lambda x: expon.pdf(x-epsilon_a, scale=exp_params['scale']))
    
    # Find optimal epsilons for exponential distribution
    opt_epsilon_exp1, max_welfare_exp1 = find_optimal_epsilon(N, epsilon_a, epsilon_b, n, c, 1, lambda x: expon.pdf(x-epsilon_a, scale=exp_params['scale']))
    opt_epsilon_exp2, max_welfare_exp2 = find_optimal_epsilon(N, epsilon_a, epsilon_b, n, c, 2, lambda x: expon.pdf(x-epsilon_a, scale=exp_params['scale']))
    
    # Print results for exponential distribution
    print("\nOptimal Epsilon Values (Exponential Distribution):")
    print(f"Equation 1: ε = {opt_epsilon_exp1:.8f}, W = {max_welfare_exp1:.8f}")
    print(f"Equation 2: ε = {opt_epsilon_exp2:.8f}, W = {max_welfare_exp2:.8f}")
    print(f"Difference: {abs(opt_epsilon_exp1 - opt_epsilon_exp2):.8f}")
    
    # Plot exponential distribution results
    plot_single_W(epsilon_values, W_exp1, opt_epsilon_exp1, "W1 for Exponential Distribution", color='b')
    plot_single_W(epsilon_values, W_exp2, opt_epsilon_exp2, "W2 for Exponential Distribution", color='r')
    
    # Print summary of all distributions
    print("\n=== Summary of All Distributions ===")
    print("Uniform Distribution:")
    print(f"  W1 Optimal ε: {opt_epsilon_uniform1:.8f}")
    print(f"  W2 Optimal ε: {opt_epsilon_uniform2:.8f}")
    print("\nNormal Distribution:")
    print(f"  W1 Optimal ε: {opt_epsilon_normal1:.8f}")
    print(f"  W2 Optimal ε: {opt_epsilon_normal2:.8f}")
    print("\nExponential Distribution:")
    print(f"  W1 Optimal ε: {opt_epsilon_exp1:.8f}")
    print(f"  W2 Optimal ε: {opt_epsilon_exp2:.8f}")
