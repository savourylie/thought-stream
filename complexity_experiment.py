"""
Systematic experiment to test if Neural ODE complexity improves performance
Start simple, add complexity step by step, log all results
"""

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from sentence_transformers import SentenceTransformer
from scipy.interpolate import interp1d
import json
import time
from datetime import datetime


class ExperimentLogger:
    """Log all experiment results for analysis"""
    
    def __init__(self):
        self.results = []
    
    def log_experiment(self, name, description, metrics, config):
        result = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'description': description,
            'metrics': metrics,
            'config': config
        }
        self.results.append(result)
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"Description: {description}")
        print(f"Config: {config}")
        print(f"Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    def save_results(self, filename="complexity_experiment_results.json"):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved results to {filename}")
    
    def compare_to_baseline(self, metrics, baseline_metrics):
        print(f"Comparison to spline baseline:")
        for key in metrics:
            if key in baseline_metrics:
                improvement = metrics[key] - baseline_metrics[key] if 'accuracy' in key else baseline_metrics[key] - metrics[key]
                print(f"  {key}: {improvement:+.4f}")


# Neural ODE Variants (increasing complexity)

class LinearODE(nn.Module):
    """Simplest: dy/dt = Ay (linear dynamics)"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain=0.01)
    
    def forward(self, t, y):
        return self.linear(y)


class BiasedLinearODE(nn.Module):
    """Add bias: dy/dt = Ay + b"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        nn.init.xavier_normal_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, t, y):
        return self.linear(y)


class TimeAwareODE(nn.Module):
    """Add time dependence: dy/dt = f(y, t)"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim + 1, dim)  # +1 for time
        nn.init.xavier_normal_(self.net.weight, gain=0.01)
        nn.init.zeros_(self.net.bias)
    
    def forward(self, t, y):
        batch_size = y.shape[0]
        t_expanded = t.expand(batch_size, 1)
        input_data = torch.cat([y, t_expanded], dim=1)
        return self.net(input_data)


class ShallowNonlinearODE(nn.Module):
    """Add single nonlinearity: dy/dt = tanh(Ay + b)"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, t, y):
        return torch.tanh(self.linear(y))


class DeepNonlinearODE(nn.Module):
    """Deeper network: dy/dt = MLP(y, t)"""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, y):
        batch_size = y.shape[0]
        t_expanded = t.expand(batch_size, 1)
        input_data = torch.cat([y, t_expanded], dim=1)
        return self.net(input_data)


class NeuralODEWrapper:
    """Wrapper to train and evaluate different ODE functions"""
    
    def __init__(self, ode_func, lr=0.001):
        self.ode_func = ode_func
        self.optimizer = torch.optim.Adam(ode_func.parameters(), lr=lr)
    
    def fit(self, embeddings, epochs=100, verbose=False):
        """Train the Neural ODE"""
        embeddings_tensor = torch.FloatTensor(embeddings)
        t_data = torch.linspace(0, 1, len(embeddings))
        
        losses = []
        start_time = time.time()
        
        self.ode_func.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            y0 = embeddings_tensor[0:1]
            pred_trajectory = odeint(self.ode_func, y0, t_data).squeeze(1)
            
            loss = nn.MSELoss()(pred_trajectory, embeddings_tensor)
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and epoch % 25 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")
        
        self.ode_func.eval()
        training_time = time.time() - start_time
        
        return {
            'final_loss': losses[-1],
            'training_time': training_time,
            'convergence_rate': (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
        }
    
    def evaluate(self, embeddings, words):
        """Evaluate reconstruction quality"""
        embeddings_tensor = torch.FloatTensor(embeddings)
        t_data = torch.linspace(0, 1, len(embeddings))
        
        with torch.no_grad():
            y0 = embeddings_tensor[0:1]
            pred_trajectory = odeint(self.ode_func, y0, t_data).squeeze(1)
        
        # Calculate reconstruction errors
        reconstruction_errors = []
        for i in range(len(embeddings)):
            error = torch.norm(pred_trajectory[i] - embeddings_tensor[i]).item()
            reconstruction_errors.append(error)
        
        # Word quantization test
        model = SentenceTransformer('all-MiniLM-L6-v2')
        vocab = words + ["dog", "cat", "mouse", "chair", "table", "floor"]
        vocab_embeddings = model.encode(vocab)
        
        quantized_words = []
        for pred_emb in pred_trajectory:
            distances = [torch.norm(pred_emb - torch.FloatTensor(vocab_emb)).item() 
                        for vocab_emb in vocab_embeddings]
            closest_idx = np.argmin(distances)
            quantized_words.append(vocab[closest_idx])
        
        word_accuracy = sum(1 for i, w in enumerate(quantized_words) 
                           if w.lower() == words[i].lower()) / len(words)
        
        return {
            'max_reconstruction_error': max(reconstruction_errors),
            'mean_reconstruction_error': np.mean(reconstruction_errors),
            'word_accuracy': word_accuracy,
            'reconstruction_errors': reconstruction_errors,
            'quantized_words': quantized_words
        }


def run_complexity_experiment():
    """Run systematic complexity experiment"""
    
    logger = ExperimentLogger()
    
    # Load test data
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence = "The cat sat on the mat"
    words = sentence.split()
    embeddings = model.encode(words)
    
    print(f"Running complexity experiment on: '{sentence}'")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Baseline: Spline interpolation
    print(f"\n--- BASELINE: Spline Interpolation ---")
    t_data = np.linspace(0, 1, len(words))
    
    spline_funcs = []
    for dim in range(embeddings.shape[1]):
        spline_funcs.append(interp1d(t_data, embeddings[:, dim], kind='cubic'))
    
    def sample_spline(t_vals):
        result = np.zeros((len(t_vals), embeddings.shape[1]))
        for dim, func in enumerate(spline_funcs):
            result[:, dim] = func(t_vals)
        return result
    
    spline_recon = sample_spline(t_data)
    spline_errors = [np.linalg.norm(spline_recon[i] - embeddings[i]) for i in range(len(words))]
    
    # Quantize spline results
    vocab = words + ["dog", "cat", "mouse", "chair", "table", "floor"] 
    vocab_embeddings = model.encode(vocab)
    
    spline_quantized = []
    for emb in spline_recon:
        distances = [np.linalg.norm(emb - vocab_emb) for vocab_emb in vocab_embeddings]
        closest_idx = np.argmin(distances)
        spline_quantized.append(vocab[closest_idx])
    
    spline_accuracy = sum(1 for i, w in enumerate(spline_quantized) 
                         if w.lower() == words[i].lower()) / len(words)
    
    baseline_metrics = {
        'max_reconstruction_error': max(spline_errors),
        'mean_reconstruction_error': np.mean(spline_errors),
        'word_accuracy': spline_accuracy
    }
    
    logger.log_experiment(
        "Baseline", 
        "Cubic spline interpolation through word embeddings",
        baseline_metrics,
        {'method': 'scipy.interp1d', 'kind': 'cubic'}
    )
    
    # Neural ODE experiments (increasing complexity)
    experiments = [
        ("Linear ODE", "dy/dt = Ay", LinearODE(384)),
        ("Linear + Bias ODE", "dy/dt = Ay + b", BiasedLinearODE(384)),
        ("Time-Aware ODE", "dy/dt = f(y,t)", TimeAwareODE(384)),
        ("Shallow Nonlinear", "dy/dt = tanh(Ay+b)", ShallowNonlinearODE(384)),
        ("Deep Nonlinear", "dy/dt = MLP(y,t)", DeepNonlinearODE(384, 64)),
    ]
    
    for name, description, ode_func in experiments:
        print(f"\n--- {name.upper()} ---")
        
        wrapper = NeuralODEWrapper(ode_func)
        
        # Training
        train_metrics = wrapper.fit(embeddings, epochs=100, verbose=True)
        
        # Evaluation
        eval_metrics = wrapper.evaluate(embeddings, words)
        
        # Combine metrics
        all_metrics = {**train_metrics, **eval_metrics}
        
        # Log experiment
        logger.log_experiment(
            name,
            description,
            all_metrics,
            {
                'architecture': ode_func.__class__.__name__,
                'parameters': sum(p.numel() for p in ode_func.parameters()),
                'epochs': 100
            }
        )
        
        # Compare to baseline
        logger.compare_to_baseline(eval_metrics, baseline_metrics)
        
        print(f"Quantized result: {eval_metrics['quantized_words']}")
    
    # Save all results
    logger.save_results()
    
    # Summary analysis
    print(f"\n{'='*60}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*60}")
    
    print(f"Baseline (Spline):")
    print(f"  Word accuracy: {baseline_metrics['word_accuracy']:.1%}")
    print(f"  Max error: {baseline_metrics['max_reconstruction_error']:.6f}")
    
    print(f"\nNeural ODE Results:")
    for result in logger.results[1:]:  # Skip baseline
        name = result['name']
        acc = result['metrics']['word_accuracy']
        error = result['metrics']['max_reconstruction_error']
        params = result['config']['parameters']
        print(f"  {name:20s}: {acc:.1%} accuracy, {error:.6f} max error, {params:5d} params")
    
    return logger.results


if __name__ == "__main__":
    results = run_complexity_experiment()