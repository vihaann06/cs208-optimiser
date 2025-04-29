import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar

def calculate_W(epsilon, epsilon_a, epsilon_b, N):
    """
    Calculate the welfare function W for given parameters.
    
    Args:
        epsilon (float): The privacy parameter to evaluate
        epsilon_a (float): Lower bound of epsilon distribution
        epsilon_b (float): Upper bound of epsilon distribution
        M (int): Total number of potential participants
    
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
    
    # Define the function for the first term's outer integral
    def first_term_integrand(x):
        return (1/x) * (1 - 1/(N * epsilon)) * inner_result
    
    # Define the function for the second term's integral
    def second_term_integrand(x):
        return 1 - epsilon/x
    
    # Calculate the first term (double integral)
    first_term = integrate.quad(first_term_integrand, epsilon_a, epsilon_b)[0]
    
    # Calculate the second term (single integral)
    second_term = integrate.quad(second_term_integrand, lower_bound, epsilon_b)[0]
    
    return first_term + second_term

def find_optimal_epsilon(epsilon_a, epsilon_b, M):
    """
    Find the value of epsilon that maximizes W.
    
    Args:
        epsilon_a (float): Lower bound of epsilon distribution
        epsilon_b (float): Upper bound of epsilon distribution
        M (int): Total number of potential participants
    
    Returns:
        tuple: (optimal_epsilon, maximum_W)
    """
    def objective(epsilon):
        # Handle the case where epsilon is below epsilon_a
        if epsilon < epsilon_a:
            return float('inf')  # Invalid region
        return -calculate_W(epsilon, epsilon_a, epsilon_b, M)  # Negative because we want to maximize
    
    # Find the minimum of the negative function (equivalent to finding maximum of W)
    result = minimize_scalar(objective, 
                           bounds=(epsilon_a, epsilon_b),
                           method='bounded')
    
    return result.x, -result.fun  # Return the optimal epsilon and the maximum W value

# Example usage
if __name__ == "__main__":
    # Example parameters
    epsilon_a = 0.1  # Lower bound for privacy preferences
    epsilon_b = 1.0  # Upper bound for privacy preferences
    N = 1000        # Total number of potential participants
    
    # Find optimal epsilon
    optimal_epsilon, max_welfare = find_optimal_epsilon(epsilon_a, epsilon_b, N)
    
    print(f"Optimal epsilon: {optimal_epsilon:.4f}")
    print(f"Maximum welfare: {max_welfare:.4f}")
