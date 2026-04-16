import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from typing import List, Tuple
import matplotlib.pyplot as plt


class ODEFunc(nn.Module):
    """Neural network defining the ODE dynamics: dθ/dt = f(θ, t)"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Simple linear transformation - works best based on debugging
        self.net = nn.Linear(embedding_dim, embedding_dim)
        
        # Initialize with small weights
        nn.init.xavier_normal_(self.net.weight, gain=0.01)
        nn.init.zeros_(self.net.bias)
    
    def forward(self, t, y):
        """
        Args:
            t: current time (scalar) - ignored in simple linear model
            y: current state [batch_size, embedding_dim]
        Returns:
            dy/dt: rate of change [batch_size, embedding_dim]
        """
        return self.net(y)


class NeuralODETrajectory(nn.Module):
    """Neural ODE for modeling thought trajectories through semantic space."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ode_func = ODEFunc(embedding_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # Use CPU for now to avoid MPS float64 issues with odeint
        self.device = torch.device("cpu")
        print("Using CPU (MPS has float64 compatibility issues with odeint)")
        
        self.to(self.device)
    
    def forward(self, start_embedding: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        """
        Solve ODE from start_embedding at times t_eval
        
        Args:
            start_embedding: [batch_size, embedding_dim] 
            t_eval: [num_times] - times to evaluate trajectory
        Returns:
            trajectory: [num_times, batch_size, embedding_dim]
        """
        # odeint expects [batch_size, embedding_dim] initial condition
        trajectory = odeint(self.ode_func, start_embedding, t_eval, method='dopri5')
        return trajectory
    
    def fit_trajectory(self, word_embeddings: np.ndarray, epochs: int = 100):
        """
        Train Neural ODE to fit through given word embeddings
        
        Args:
            word_embeddings: [num_words, embedding_dim] numpy array
            epochs: number of training epochs
        """
        # Convert to torch tensors and move to device
        embeddings = torch.FloatTensor(word_embeddings).to(self.device)
        num_words = len(word_embeddings)
        
        # Create time points (0 to 1)
        t_data = torch.linspace(0, 1, num_words).to(self.device)
        
        self.train()
        losses = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Start from first word (simple approach that works)
            start_state = embeddings[0:1]  # [1, embedding_dim]
            
            pred_trajectory = self(start_state, t_data)  # [num_words, 1, embedding_dim]
            pred_trajectory = pred_trajectory.squeeze(1)  # [num_words, embedding_dim]
            
            # Simple MSE loss (no complex regularization)
            loss = nn.MSELoss()(pred_trajectory, embeddings)
            
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 25 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        self.eval()
        return losses
    
    def sample_trajectory(self, start_embedding: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        """
        Sample the learned trajectory at specified time points
        
        Args:
            start_embedding: [embedding_dim] starting point
            t_eval: [num_samples] time points to sample
        Returns:
            trajectory_points: [num_samples, embedding_dim]
        """
        with torch.no_grad():
            start_tensor = torch.FloatTensor(start_embedding).unsqueeze(0).to(self.device)  # [1, embedding_dim]
            t_tensor = torch.FloatTensor(t_eval).to(self.device)
            
            trajectory = self(start_tensor, t_tensor)  # [num_samples, 1, embedding_dim]
            return trajectory.squeeze(1).cpu().numpy()  # [num_samples, embedding_dim]


def demo_neural_ode_vs_spline():
    """Compare Neural ODE trajectories with spline interpolation."""
    print("\n=== Neural ODE vs Spline Comparison ===")
    
    # Import our existing spline-based system
    import sys
    sys.path.append('.')
    from main import ThoughtTrajectory
    
    # Initialize both systems
    ts_spline = ThoughtTrajectory()
    
    # Test sentence
    sentence = "The cat sat on the mat"
    print(f"Sentence: {sentence}")
    
    # Get embeddings
    embeddings = ts_spline.encode_sentence(sentence)
    print(f"Encoded to {embeddings.shape}")
    
    # 1. Spline trajectory (our Phase 1 method)
    print("\n--- Training Spline Trajectory ---")
    spline_traj = ts_spline.fit_trajectory(embeddings)
    
    # 2. Neural ODE trajectory  
    print("\n--- Training Neural ODE Trajectory ---")
    neural_ode = NeuralODETrajectory(embedding_dim=embeddings.shape[1])
    losses = neural_ode.fit_trajectory(embeddings, epochs=100)
    
    # Sample both trajectories
    t_sample = np.linspace(0, 1, 20)  # More dense sampling for comparison
    
    spline_samples = spline_traj(t_sample)
    ode_samples = neural_ode.sample_trajectory(embeddings[0], t_sample)
    
    # Compare reconstruction quality
    vocabulary = sentence.split() + ["dog", "mouse", "chair", "table", "floor", "the", "a"]
    
    print("\n--- Reconstruction Comparison ---")
    
    # Debug: check what embeddings look like
    print(f"ODE sample distances to original words:")
    for i, ode_sample in enumerate(ode_samples[:len(embeddings)]):
        distances = []
        for j, word in enumerate(sentence.split()):
            orig_emb = embeddings[j]
            dist = np.linalg.norm(ode_sample - orig_emb)
            distances.append((dist, word))
        distances.sort()
        print(f"  Sample {i}: closest to '{distances[0][1]}' (dist: {distances[0][0]:.3f})")
    
    # Original word count reconstruction
    original_count = len(embeddings)
    spline_recon = ts_spline.reconstruct_sentence(spline_traj, original_count, vocabulary)
    ode_recon = ts_spline.quantize_to_words(ode_samples[:original_count], vocabulary)
    ode_recon_str = ' '.join(ode_recon)
    
    print(f"Original:    {sentence}")
    print(f"Spline:      {spline_recon}")  
    print(f"Neural ODE:  {ode_recon_str}")
    
    # Compressed reconstruction
    compressed_samples = 3
    spline_compressed = ts_spline.reconstruct_sentence(spline_traj, compressed_samples, vocabulary)
    ode_compressed = ts_spline.quantize_to_words(ode_samples[:compressed_samples], vocabulary)
    ode_compressed_str = ' '.join(ode_compressed)
    
    print(f"\nCompressed (3 words):")
    print(f"Spline:      {spline_compressed}")
    print(f"Neural ODE:  {ode_compressed_str}")
    
    # Trajectory smoothness comparison
    print(f"\n--- Trajectory Analysis ---")
    
    # Compute trajectory "roughness" (sum of second derivatives)
    def compute_roughness(trajectory_samples):
        if len(trajectory_samples) < 3:
            return 0
        # Approximate second derivative
        second_deriv = np.diff(trajectory_samples, n=2, axis=0)
        return np.mean(np.linalg.norm(second_deriv, axis=1))
    
    spline_roughness = compute_roughness(spline_samples)
    ode_roughness = compute_roughness(ode_samples)
    
    print(f"Spline roughness:     {spline_roughness:.6f}")
    print(f"Neural ODE roughness: {ode_roughness:.6f}")
    print(f"Smoothness improvement: {((spline_roughness - ode_roughness) / spline_roughness * 100):.1f}%")
    
    return neural_ode, losses


if __name__ == "__main__":
    demo_neural_ode_vs_spline()