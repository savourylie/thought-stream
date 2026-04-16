"""
Debug Neural ODE training step by step
Start with simple 2D case, then scale up
"""

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt


class SimpleODEFunc(nn.Module):
    """Very simple ODE function for debugging"""
    
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        # Single linear layer to start
        self.net = nn.Linear(dim, dim)
        
    def forward(self, t, y):
        """dy/dt = Ay where A is learned matrix"""
        return self.net(y)


def test_2d_trajectory():
    """Test on simple 2D trajectory we can visualize"""
    print("=== Testing 2D Neural ODE ===")
    
    # Create simple trajectory: straight line from (0,0) to (1,1)
    t_data = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    y_data = torch.tensor([
        [0.0, 0.0],
        [0.25, 0.25], 
        [0.5, 0.5],
        [0.75, 0.75],
        [1.0, 1.0]
    ])
    
    print(f"Target trajectory: {y_data}")
    
    # Initialize ODE
    ode_func = SimpleODEFunc(dim=2)
    
    # Training loop
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=0.01)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        
        # Start from first point
        y0 = y_data[0:1]  # [1, 2]
        
        # Solve ODE
        pred_y = odeint(ode_func, y0, t_data)  # [5, 1, 2]
        pred_y = pred_y.squeeze(1)  # [5, 2]
        
        # Loss
        loss = nn.MSELoss()(pred_y, y_data)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            print(f"  Predicted: {pred_y[1].detach().numpy()}")
            print(f"  Target:    {y_data[1].numpy()}")
    
    # Test final result
    with torch.no_grad():
        pred_y = odeint(ode_func, y_data[0:1], t_data).squeeze(1)
        print(f"\nFinal predictions vs targets:")
        for i, (pred, true) in enumerate(zip(pred_y, y_data)):
            print(f"  t={t_data[i]:.2f}: pred={pred.numpy()}, true={true.numpy()}")
    
    return ode_func, pred_y, y_data


def test_embedding_trajectory():
    """Test on actual word embeddings but smaller scale"""
    print("\n=== Testing Word Embedding Neural ODE ===")
    
    # Load sentence transformer
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Simple 3-word sentence
    words = ["cat", "sat", "mat"]
    embeddings = model.encode(words)
    embeddings_tensor = torch.FloatTensor(embeddings)  # [3, 384]
    
    print(f"Embeddings shape: {embeddings_tensor.shape}")
    print(f"Word distances: cat->sat {torch.norm(embeddings_tensor[1] - embeddings_tensor[0]):.3f}")
    print(f"                sat->mat {torch.norm(embeddings_tensor[2] - embeddings_tensor[1]):.3f}")
    
    # Time points
    t_data = torch.tensor([0.0, 0.5, 1.0])
    
    # Simple ODE function for high-dimensional case
    class EmbeddingODEFunc(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            # Very simple: just a linear transformation
            self.linear = nn.Linear(embed_dim, embed_dim)
            
            # Initialize to small values
            nn.init.xavier_normal_(self.linear.weight, gain=0.01)
            nn.init.zeros_(self.linear.bias)
        
        def forward(self, t, y):
            return self.linear(y)
    
    ode_func = EmbeddingODEFunc(384)
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=0.001)
    
    print(f"\nTraining Neural ODE on embeddings...")
    for epoch in range(500):
        optimizer.zero_grad()
        
        # Start from "cat"
        y0 = embeddings_tensor[0:1]  # [1, 384]
        
        # Solve ODE
        pred_trajectory = odeint(ode_func, y0, t_data)  # [3, 1, 384]
        pred_trajectory = pred_trajectory.squeeze(1)    # [3, 384]
        
        # Loss against all three words
        loss = nn.MSELoss()(pred_trajectory, embeddings_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
            # Check if we're fitting the words correctly
            with torch.no_grad():
                pred_words = []
                for i, pred_emb in enumerate(pred_trajectory):
                    # Find closest word
                    distances = [torch.norm(pred_emb - embeddings_tensor[j]) for j in range(3)]
                    closest_idx = np.argmin(distances)
                    pred_words.append(words[closest_idx])
                print(f"  Predicted words: {pred_words}")
    
    # Final test
    print(f"\nFinal test:")
    with torch.no_grad():
        pred_trajectory = odeint(ode_func, embeddings_tensor[0:1], t_data).squeeze(1)
        
        pred_words = []
        for i, pred_emb in enumerate(pred_trajectory):
            distances = [torch.norm(pred_emb - embeddings_tensor[j]) for j in range(3)]
            closest_idx = np.argmin(distances)
            pred_words.append(words[closest_idx])
            print(f"  t={t_data[i]:.1f}: {words[i]} -> {pred_words[i]} (dist: {distances[closest_idx]:.3f})")
    
    return ode_func, pred_trajectory


def main():
    print("🔧 Debugging Neural ODE Training")
    
    # Step 1: Test simple 2D case
    ode_func_2d, pred_2d, true_2d = test_2d_trajectory()
    
    # Step 2: Test word embeddings
    ode_func_emb, pred_emb = test_embedding_trajectory()
    
    print(f"\n✅ Neural ODE debugging complete!")


if __name__ == "__main__":
    main()