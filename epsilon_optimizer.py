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
        N (int): Number of participants
    
    Returns:
        float: The value of the welfare function W
    """
    def inner_integral(y):
        return 1/y
    
    def outer_integral(x):
        # Calculate the inner integral
        inner_result = integrate.quad(inner_integral, max(epsilon, epsilon_a), epsilon_b)[0]
        # Calculate the outer integrand
        return (1/x) * (1 - 1/(N * epsilon)) * inner_result
    
    # First term: double integral
    first_term = integrate.quad(outer_integral, epsilon_a, epsilon_b)[0]
    
    # Second term: single integral
    def second_integral(x):
        return 1 - epsilon/x
    
    second_term = N * integrate.quad(second_integral, epsilon, epsilon_b)[0]
    
    return first_term + second_term

def find_optimal_epsilon(epsilon_a, epsilon_b, N):
    """
    Find the value of epsilon that maximizes W.
    
    Args:
        epsilon_a (float): Lower bound of epsilon distribution
        epsilon_b (float): Upper bound of epsilon distribution
        N (int): Number of participants
    
    Returns:
        tuple: (optimal_epsilon, maximum_W)
    """
    def objective(epsilon):
        return -calculate_W(epsilon, epsilon_a, epsilon_b, N)  # Negative because we want to maximize
    
    # Find the minimum of the negative function (equivalent to finding maximum of W)
    result = minimize_scalar(objective, 
                           bounds=(epsilon_a, epsilon_b),
                           method='bounded')
    
    return result.x, -result.fun  # Return the optimal epsilon and the maximum W value

if __name__ == "__main__":
    # Example usage with the given parameters
    epsilon_a = 0.01
    epsilon_b = 2.0
    N = 100
    
    optimal_epsilon, max_welfare = find_optimal_epsilon(epsilon_a, epsilon_b, N)
    print(f"Optimal epsilon: {optimal_epsilon:.6f}")
    print(f"Maximum welfare W: {max_welfare:.6f}")
    
    # Print some additional points to verify the maximum
    test_points = np.linspace(epsilon_a, epsilon_b, 5)
    print("\nWelfare values at different points:")
    for eps in test_points:
        w = calculate_W(eps, epsilon_a, epsilon_b, N)
        print(f"epsilon = {eps:.3f}, W = {w:.6f}") 