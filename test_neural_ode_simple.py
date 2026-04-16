"""Test if Neural ODE is actually learning correctly"""

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from sentence_transformers import SentenceTransformer


class SimpleODEFunc(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Linear(embedding_dim, embedding_dim)
        nn.init.xavier_normal_(self.net.weight, gain=0.01)
        nn.init.zeros_(self.net.bias)
    
    def forward(self, t, y):
        return self.net(y)


def test_sentence_neural_ode():
    print("=== Testing Neural ODE on sentence ===")
    
    # Load sentence
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence = "The cat sat on the mat"
    words = sentence.split()
    embeddings = model.encode(words)
    embeddings_tensor = torch.FloatTensor(embeddings)
    
    print(f"Words: {words}")
    print(f"Embeddings shape: {embeddings_tensor.shape}")
    
    # Time points
    t_data = torch.linspace(0, 1, len(words))
    
    # Neural ODE
    ode_func = SimpleODEFunc(384)
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=0.001)
    
    print("Training...")
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Start from first word
        y0 = embeddings_tensor[0:1]
        pred_trajectory = odeint(ode_func, y0, t_data)
        pred_trajectory = pred_trajectory.squeeze(1)
        
        loss = nn.MSELoss()(pred_trajectory, embeddings_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Test reconstruction
    print("\nTesting reconstruction...")
    with torch.no_grad():
        pred_trajectory = odeint(ode_func, embeddings_tensor[0:1], t_data)
        pred_trajectory = pred_trajectory.squeeze(1)
        
        print("Distances from predicted to actual embeddings:")
        for i, (word, pred_emb, true_emb) in enumerate(zip(words, pred_trajectory, embeddings_tensor)):
            dist = torch.norm(pred_emb - true_emb).item()
            print(f"  {word}: {dist:.4f}")
        
        # Test quantization directly
        print("\nQuantization test (which word is closest?):")
        for i, (word, pred_emb) in enumerate(zip(words, pred_trajectory)):
            distances = []
            for j, true_emb in enumerate(embeddings_tensor):
                dist = torch.norm(pred_emb - true_emb).item()
                distances.append((dist, words[j]))
            
            distances.sort()
            closest_word = distances[0][1]
            closest_dist = distances[0][0]
            
            print(f"  Predicted embedding {i} -> '{closest_word}' (dist: {closest_dist:.4f})")


if __name__ == "__main__":
    test_sentence_neural_ode()