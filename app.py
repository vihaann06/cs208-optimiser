from flask import Flask, render_template, request, jsonify
import sys
import os

# Import the epsilon_optimizer module directly
from epsilon_optimizer import find_optimal_epsilon

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/calculate_optimal", methods=['POST'])
def calculate_optimal():
    data = request.json
    epsilon_a = float(data.get('epsilon_a', 0.01))
    epsilon_b = float(data.get('epsilon_b', 2.0))
    N = int(data.get('N', 100))
    
    try:
        optimal_epsilon, max_welfare = find_optimal_epsilon(epsilon_a, epsilon_b, N)
        return jsonify({
            'optimal_epsilon': optimal_epsilon,
            'max_welfare': max_welfare
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5001) 