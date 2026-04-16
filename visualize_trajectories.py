"""
Create clear visualizations comparing spline vs Neural ODE trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sentence_transformers import SentenceTransformer
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import seaborn as sns

plt.style.use('seaborn-v0_8')


def create_trajectory_comparison():
    """Compare spline vs Neural ODE trajectories with proper visualization"""
    
    print("=== Creating Trajectory Visualization ===")
    
    # Load sentence transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test sentence
    sentence = "The cat sat on the mat"
    words = sentence.split()
    embeddings = model.encode(words)
    
    print(f"Sentence: {sentence}")
    print(f"Words: {words}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Reduce dimensionality to 2D for visualization using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Time points for original words
    t_original = np.linspace(0, 1, len(words))
    
    # Method 1: Spline interpolation
    spline_x = interp1d(t_original, embeddings_2d[:, 0], kind='cubic')
    spline_y = interp1d(t_original, embeddings_2d[:, 1], kind='cubic')
    
    # Method 2: Neural ODE
    class SimpleODEFunc(nn.Module):
        def __init__(self, dim=384):
            super().__init__()
            self.net = nn.Linear(dim, dim)
            nn.init.xavier_normal_(self.net.weight, gain=0.01)
            nn.init.zeros_(self.net.bias)
        
        def forward(self, t, y):
            return self.net(y)
    
    # Train Neural ODE on full embeddings
    print("Training Neural ODE...")
    ode_func = SimpleODEFunc(384)
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=0.001)
    embeddings_tensor = torch.FloatTensor(embeddings)
    t_tensor = torch.FloatTensor(t_original)
    
    for epoch in range(100):
        optimizer.zero_grad()
        y0 = embeddings_tensor[0:1]
        pred_trajectory = odeint(ode_func, y0, t_tensor).squeeze(1)
        loss = nn.MSELoss()(pred_trajectory, embeddings_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Sample both trajectories densely
    t_dense = np.linspace(0, 1, 100)
    
    # Spline samples in 2D
    spline_samples_2d = np.column_stack([spline_x(t_dense), spline_y(t_dense)])
    
    # Neural ODE samples in full space, then project to 2D
    with torch.no_grad():
        t_dense_tensor = torch.FloatTensor(t_dense)
        ode_samples = odeint(ode_func, embeddings_tensor[0:1], t_dense_tensor).squeeze(1).numpy()
    ode_samples_2d = pca.transform(ode_samples)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Original word embeddings in 2D
    ax1 = axes[0, 0]
    scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=range(len(words)), s=100, cmap='viridis', alpha=0.8)
    for i, word in enumerate(words):
        ax1.annotate(f'{i}: {word}', 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax1.set_title('Original Word Embeddings (2D PCA)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    
    # Plot 2: Spline trajectory
    ax2 = axes[0, 1] 
    ax2.plot(spline_samples_2d[:, 0], spline_samples_2d[:, 1], 'b-', 
            linewidth=2, label='Spline Trajectory', alpha=0.7)
    ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
               c=range(len(words)), s=100, cmap='viridis', alpha=0.8)
    for i, word in enumerate(words):
        ax2.annotate(f'{word}', 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax2.set_title('Spline Interpolation Trajectory')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    
    # Plot 3: Neural ODE trajectory
    ax3 = axes[1, 0]
    ax3.plot(ode_samples_2d[:, 0], ode_samples_2d[:, 1], 'm-', 
            linewidth=2, label='Neural ODE Trajectory', alpha=0.7)
    ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
               c=range(len(words)), s=100, cmap='viridis', alpha=0.8)
    for i, word in enumerate(words):
        ax3.annotate(f'{word}', 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax3.set_title('Neural ODE Trajectory')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    
    # Plot 4: Both trajectories together
    ax4 = axes[1, 1]
    ax4.plot(spline_samples_2d[:, 0], spline_samples_2d[:, 1], 'b-', 
            linewidth=2, label='Spline', alpha=0.7)
    ax4.plot(ode_samples_2d[:, 0], ode_samples_2d[:, 1], 'm-', 
            linewidth=2, label='Neural ODE', alpha=0.7)
    ax4.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
               c=range(len(words)), s=100, cmap='viridis', alpha=0.8, zorder=5)
    for i, word in enumerate(words):
        ax4.annotate(f'{word}', 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax4.set_title('Comparison: Spline vs Neural ODE')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison_detailed.png', dpi=150, bbox_inches='tight')
    print("Saved trajectory_comparison_detailed.png")
    plt.show()
    
    # Quantitative comparison
    print(f"\n=== Quantitative Analysis ===")
    
    # Calculate path lengths
    def path_length(trajectory):
        diffs = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)
    
    spline_length = path_length(spline_samples_2d)
    ode_length = path_length(ode_samples_2d)
    
    print(f"Spline path length: {spline_length:.3f}")
    print(f"Neural ODE path length: {ode_length:.3f}")
    print(f"Length ratio (ODE/Spline): {ode_length/spline_length:.3f}")
    
    # Calculate trajectory smoothness (curvature)
    def trajectory_roughness(trajectory):
        if len(trajectory) < 3:
            return 0
        second_deriv = np.diff(trajectory, n=2, axis=0)
        return np.mean(np.linalg.norm(second_deriv, axis=1))
    
    spline_roughness = trajectory_roughness(spline_samples_2d)
    ode_roughness = trajectory_roughness(ode_samples_2d)
    
    print(f"Spline roughness: {spline_roughness:.6f}")
    print(f"Neural ODE roughness: {ode_roughness:.6f}")
    print(f"Smoothness improvement: {((spline_roughness - ode_roughness) / spline_roughness * 100):.1f}%")
    
    # Test reconstruction at original time points
    print(f"\n=== Reconstruction Quality ===")
    
    # Sample both trajectories at original time points  
    spline_at_words = np.column_stack([spline_x(t_original), spline_y(t_original)])
    
    with torch.no_grad():
        ode_at_words = odeint(ode_func, embeddings_tensor[0:1], t_tensor).squeeze(1).numpy()
    ode_at_words_2d = pca.transform(ode_at_words)
    
    print("Reconstruction errors (in 2D PCA space):")
    for i, word in enumerate(words):
        spline_error = np.linalg.norm(spline_at_words[i] - embeddings_2d[i])
        ode_error = np.linalg.norm(ode_at_words_2d[i] - embeddings_2d[i])
        
        print(f"  {word}: Spline={spline_error:.4f}, ODE={ode_error:.4f}")
    
    return {
        'embeddings_2d': embeddings_2d,
        'spline_samples_2d': spline_samples_2d, 
        'ode_samples_2d': ode_samples_2d,
        'words': words
    }


if __name__ == "__main__":
    results = create_trajectory_comparison()