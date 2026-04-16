"""
Learnable Spline Coefficients for Semantic Trajectory Modeling

Step 1: Match naive spline performance
Step 2: Add semantic awareness 
Step 3: Test on complex sentences
"""

import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import List, Tuple
import json
import nltk
from nltk.corpus import brown


class LearnableSplines(nn.Module):
    """Learnable spline parameters for semantic-aware trajectory modeling"""
    
    def __init__(self, num_words: int, embedding_dim: int):
        super().__init__()
        self.num_words = num_words
        self.embedding_dim = embedding_dim
        
        # Core idea: Learn how to modify cubic spline behavior
        # Start with parameters that should reproduce standard cubic splines
        
        # Tension parameters: control spline tightness between word pairs
        self.tension_params = nn.Parameter(torch.ones(num_words - 1))
        
        # Semantic weights: different importance for different embedding dimensions
        self.semantic_weights = nn.Parameter(torch.ones(embedding_dim))
        
        # Bias terms: allow slight adjustments to word embeddings
        self.word_biases = nn.Parameter(torch.zeros(num_words, embedding_dim))
        
        # Curvature controls: how much curvature allowed at each word
        self.curvature_controls = nn.Parameter(torch.ones(num_words))
        
    def forward(self, word_embeddings: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
        """
        Interpolate through word embeddings with learned parameters
        
        Args:
            word_embeddings: [num_words, embedding_dim] - input word embeddings
            t_query: [num_samples] - time points to sample (0 to 1)
            
        Returns:
            trajectory: [num_samples, embedding_dim] - interpolated trajectory
        """
        # Apply learned biases to word embeddings
        adjusted_embeddings = word_embeddings + self.word_biases
        
        # Apply semantic weights (emphasize important dimensions)
        weighted_embeddings = adjusted_embeddings * self.semantic_weights.unsqueeze(0)
        
        # Create learnable spline interpolation
        return self._learnable_interpolate(weighted_embeddings, t_query)
    
    def _learnable_interpolate(self, embeddings: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
        """
        Custom spline interpolation with learned parameters
        
        This is the key innovation: instead of fixed cubic splines,
        learn tension, curvature, and semantic weighting
        """
        num_samples = len(t_query)
        result = torch.zeros(num_samples, self.embedding_dim)
        
        # Time points for original words
        t_words = torch.linspace(0, 1, self.num_words)
        
        for i, t in enumerate(t_query):
            # Find which segment this t falls into
            if t <= 0:
                result[i] = embeddings[0]
            elif t >= 1:
                result[i] = embeddings[-1]
            else:
                # Find segment
                segment_idx = 0
                for j in range(self.num_words - 1):
                    if t_words[j] <= t <= t_words[j + 1]:
                        segment_idx = j
                        break
                
                # Local interpolation parameter (0 to 1 within segment)
                t_local = (t - t_words[segment_idx]) / (t_words[segment_idx + 1] - t_words[segment_idx])
                
                # Get segment points
                p0 = embeddings[max(0, segment_idx - 1)]
                p1 = embeddings[segment_idx]
                p2 = embeddings[segment_idx + 1]
                p3 = embeddings[min(self.num_words - 1, segment_idx + 2)]
                
                # Learnable cubic interpolation
                tension = torch.sigmoid(self.tension_params[segment_idx])  # 0-1 range
                
                # Apply curvature controls
                c1 = self.curvature_controls[segment_idx] 
                c2 = self.curvature_controls[segment_idx + 1]
                
                # Modified Catmull-Rom spline with learned parameters
                result[i] = self._cubic_interpolate(p0, p1, p2, p3, t_local, tension, c1, c2)
        
        return result
    
    def _cubic_interpolate(self, p0, p1, p2, p3, t, tension, c1, c2):
        """
        Cubic interpolation with learnable parameters
        
        Standard Catmull-Rom: interpolate between p1 and p2 using p0, p3 for derivatives
        Learnable: modify with tension, curvature controls
        """
        t2 = t * t
        t3 = t2 * t
        
        # Standard Catmull-Rom basis functions
        v0 = -0.5 * t3 + t2 - 0.5 * t
        v1 = 1.5 * t3 - 2.5 * t2 + 1.0
        v2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
        v3 = 0.5 * t3 - 0.5 * t2
        
        # Apply tension and curvature learning
        # Tension: 0=linear, 1=full cubic
        tension_factor = tension
        v0 = v0 * tension_factor
        v3 = v3 * tension_factor
        
        # Curvature controls at endpoints
        v1 = v1 * c1
        v2 = v2 * c2
        
        # Normalize to maintain interpolation
        total = v0 + v1 + v2 + v3
        v0, v1, v2, v3 = v0/total, v1/total, v2/total, v3/total
        
        return v0 * p0 + v1 * p1 + v2 * p2 + v3 * p3


class LearnableSplineTrajectory:
    """Wrapper class for training and evaluating learnable splines"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.model = None
        
    def fit(self, word_embeddings: np.ndarray, epochs: int = 200, lr: float = 0.01, verbose: bool = True):
        """Train learnable spline to match word embeddings"""
        
        embeddings_tensor = torch.FloatTensor(word_embeddings)
        num_words = len(word_embeddings)
        
        # Initialize model
        self.model = LearnableSplines(num_words, self.embedding_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Time points for training
        t_train = torch.linspace(0, 1, num_words)
        
        losses = []
        
        if verbose:
            print(f"Training learnable splines on {num_words} words...")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            pred_trajectory = self.model(embeddings_tensor, t_train)
            
            # Loss: reconstruction + regularization
            reconstruction_loss = nn.MSELoss()(pred_trajectory, embeddings_tensor)
            
            # Regularization: keep parameters reasonable
            tension_reg = torch.mean((self.model.tension_params - 1.0) ** 2) * 0.01
            semantic_reg = torch.mean((self.model.semantic_weights - 1.0) ** 2) * 0.01
            bias_reg = torch.mean(self.model.word_biases ** 2) * 0.001
            curvature_reg = torch.mean((self.model.curvature_controls - 1.0) ** 2) * 0.01
            
            total_loss = reconstruction_loss + tension_reg + semantic_reg + bias_reg + curvature_reg
            
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            
            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch:3d}, Loss: {total_loss.item():.6f}, Recon: {reconstruction_loss.item():.6f}")
        
        if verbose:
            print(f"Training completed. Final loss: {losses[-1]:.6f}")
        
        return losses
    
    def sample_trajectory(self, word_embeddings: np.ndarray, t_sample: np.ndarray) -> np.ndarray:
        """Sample the learned trajectory at specified time points"""
        
        if self.model is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        embeddings_tensor = torch.FloatTensor(word_embeddings)
        t_tensor = torch.FloatTensor(t_sample)
        
        with torch.no_grad():
            trajectory = self.model(embeddings_tensor, t_tensor)
        
        return trajectory.numpy()
    
    def get_learned_parameters(self):
        """Get learned parameters for analysis"""
        if self.model is None:
            return None
        
        return {
            'tension_params': torch.sigmoid(self.model.tension_params).detach().numpy(),
            'semantic_weights': self.model.semantic_weights.detach().numpy(),
            'word_biases': self.model.word_biases.detach().numpy(),
            'curvature_controls': self.model.curvature_controls.detach().numpy()
        }


def test_basic_performance():
    """Test 1: Match naive spline performance on simple sentence"""
    
    print("=== TEST 1: Basic Performance vs Naive Splines ===")
    
    # Load test sentence
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence = "The cat sat on the mat"
    words = sentence.split()
    embeddings = model.encode(words)
    
    print(f"Test sentence: {sentence}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Baseline: naive splines
    print("\n--- Naive Cubic Splines (Baseline) ---")
    t_original = np.linspace(0, 1, len(words))
    
    spline_funcs = []
    for dim in range(embeddings.shape[1]):
        spline_funcs.append(interp1d(t_original, embeddings[:, dim], kind='cubic'))
    
    def sample_naive_spline(t_vals):
        result = np.zeros((len(t_vals), embeddings.shape[1]))
        for dim, func in enumerate(spline_funcs):
            result[:, dim] = func(t_vals)
        return result
    
    naive_recon = sample_naive_spline(t_original)
    naive_errors = [np.linalg.norm(naive_recon[i] - embeddings[i]) for i in range(len(words))]
    
    print(f"Naive spline reconstruction errors: {[f'{e:.6f}' for e in naive_errors]}")
    print(f"Naive spline max error: {max(naive_errors):.8f}")
    
    # Learnable splines
    print("\n--- Learnable Splines ---")
    learnable_spline = LearnableSplineTrajectory(embeddings.shape[1])
    losses = learnable_spline.fit(embeddings, epochs=200, lr=0.01, verbose=True)
    
    learnable_recon = learnable_spline.sample_trajectory(embeddings, t_original)
    learnable_errors = [np.linalg.norm(learnable_recon[i] - embeddings[i]) for i in range(len(words))]
    
    print(f"Learnable spline reconstruction errors: {[f'{e:.6f}' for e in learnable_errors]}")
    print(f"Learnable spline max error: {max(learnable_errors):.8f}")
    
    # Quantization test
    print("\n--- Quantization Test ---")
    vocab = words + ["dog", "mouse", "chair", "table", "floor"]
    vocab_embeddings = model.encode(vocab)
    
    def quantize_trajectory(trajectory_embeddings):
        quantized = []
        for emb in trajectory_embeddings:
            distances = [np.linalg.norm(emb - vocab_emb) for vocab_emb in vocab_embeddings]
            closest_idx = np.argmin(distances)
            quantized.append(vocab[closest_idx])
        return quantized
    
    naive_words = quantize_trajectory(naive_recon)
    learnable_words = quantize_trajectory(learnable_recon)
    
    naive_accuracy = sum(1 for i, w in enumerate(naive_words) if w.lower() == words[i].lower()) / len(words)
    learnable_accuracy = sum(1 for i, w in enumerate(learnable_words) if w.lower() == words[i].lower()) / len(words)
    
    print(f"Original:           {words}")
    print(f"Naive splines:      {naive_words} (Accuracy: {naive_accuracy:.1%})")
    print(f"Learnable splines:  {learnable_words} (Accuracy: {learnable_accuracy:.1%})")
    
    # Success criteria
    success_criteria = [
        ("Max reconstruction error < 1e-3", max(learnable_errors) < 1e-3),
        ("Word accuracy = 100%", learnable_accuracy == 1.0),
        ("Match or beat naive splines", max(learnable_errors) <= max(naive_errors) * 1.1)  # Allow 10% tolerance
    ]
    
    print(f"\n--- Success Evaluation ---")
    for criterion, passed in success_criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {criterion}: {status}")
    
    overall_success = all(passed for _, passed in success_criteria)
    
    if overall_success:
        print(f"\n🎉 SUCCESS! Learnable splines match naive spline performance!")
    else:
        print(f"\n❌ Need improvement. Issues:")
        for criterion, passed in success_criteria:
            if not passed:
                print(f"   - {criterion}")
    
    # Show learned parameters
    print(f"\n--- Learned Parameters ---")
    params = learnable_spline.get_learned_parameters()
    if params:
        print(f"Tension params: {params['tension_params']}")
        print(f"Semantic weights (first 10): {params['semantic_weights'][:10]}")
        print(f"Max word bias magnitude: {np.max(np.abs(params['word_biases'])):.6f}")
        print(f"Curvature controls: {params['curvature_controls']}")
    
    return {
        'naive_errors': naive_errors,
        'learnable_errors': learnable_errors,
        'naive_accuracy': naive_accuracy,
        'learnable_accuracy': learnable_accuracy,
        'overall_success': overall_success,
        'learned_params': params
    }


def create_complex_test_cases():
    """Create challenging sentences that should benefit from semantic awareness"""
    
    # Categories of complex sentences that naive splines should struggle with
    test_cases = [
        {
            'category': 'Semantic Transitions',
            'sentence': 'The tiny kitten transformed into a massive lion',
            'challenge': 'cat->feline transition should be smoother than random words',
            'expected_advantage': 'Learnable splines should recognize semantic similarity'
        },
        {
            'category': 'Technical to Simple',
            'sentence': 'Machine learning algorithms enable predictive analytics',
            'challenge': 'Technical terms have different embedding densities',
            'expected_advantage': 'Adaptive tension for technical vs common words'
        },
        {
            'category': 'Emotional Progression',
            'sentence': 'I was devastated but eventually became joyful',
            'challenge': 'Emotion words form clusters in embedding space',
            'expected_advantage': 'Smooth emotional trajectory paths'
        },
        {
            'category': 'Abstract to Concrete',
            'sentence': 'Democracy requires active citizen participation in elections',
            'challenge': 'Abstract concepts vs concrete actions',
            'expected_advantage': 'Different curvature for abstract vs concrete'
        },
        {
            'category': 'Scientific Progression',
            'sentence': 'Photons exhibit wave particle duality in quantum mechanics',
            'challenge': 'Scientific concept progression',
            'expected_advantage': 'Physics concepts should flow smoothly'
        },
        {
            'category': 'Narrative Flow',
            'sentence': 'The detective investigated clues and solved the mystery',
            'challenge': 'Logical narrative progression',
            'expected_advantage': 'Story-aware trajectory smoothness'
        }
    ]
    
    return test_cases


class SemanticAwareLearning:
    """Enhanced learnable splines with semantic awareness"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        
    def compute_semantic_similarity_matrix(self, word_embeddings: np.ndarray):
        """Compute pairwise semantic similarities between words"""
        n_words = len(word_embeddings)
        similarity_matrix = np.zeros((n_words, n_words))
        
        for i in range(n_words):
            for j in range(n_words):
                # Cosine similarity
                cos_sim = np.dot(word_embeddings[i], word_embeddings[j]) / (
                    np.linalg.norm(word_embeddings[i]) * np.linalg.norm(word_embeddings[j])
                )
                similarity_matrix[i, j] = cos_sim
        
        return similarity_matrix
    
    def semantic_tension_loss(self, model, word_embeddings):
        """Loss that encourages semantic-aware tension parameters"""
        
        similarity_matrix = self.compute_semantic_similarity_matrix(word_embeddings.detach().numpy())
        
        # Get learned tension parameters (0-1 after sigmoid)
        tensions = torch.sigmoid(model.tension_params)
        
        # Idea: High similarity between adjacent words should mean low tension (more linear)
        # Low similarity should mean high tension (more cubic curvature)
        semantic_loss = 0
        
        for i in range(len(tensions)):
            # Similarity between adjacent words
            adj_similarity = similarity_matrix[i, i + 1]
            
            # Convert similarity to desired tension
            # High similarity (0.9) -> low tension (0.2)
            # Low similarity (0.3) -> high tension (0.8) 
            desired_tension = 1.0 - adj_similarity  # Invert similarity
            
            # Loss if tension deviates from semantic expectation
            semantic_loss += (tensions[i] - desired_tension) ** 2
        
        return semantic_loss / len(tensions)
    
    def embedding_density_regularization(self, model, word_embeddings):
        """Regularization based on local embedding density"""
        
        n_words = len(word_embeddings)
        density_reg = 0
        
        for i in range(n_words):
            # Compute local density (distance to nearest neighbors)
            distances = []
            for j in range(n_words):
                if i != j:
                    dist = torch.norm(word_embeddings[i] - word_embeddings[j])
                    distances.append(dist)
            
            # Local density = 1 / mean distance to neighbors
            mean_dist = torch.mean(torch.stack(distances))
            local_density = 1.0 / (mean_dist + 1e-8)
            
            # High density words should have less bias (they're well-represented)
            # Low density words can have more bias (they're outliers)
            bias_magnitude = torch.norm(model.word_biases[i])
            
            # Penalty if high-density words have large biases
            density_reg += local_density * bias_magnitude ** 2
        
        return density_reg / n_words


def test_semantic_advantages():
    """Test 2: Show learnable splines outperform naive splines on complex sentences"""
    
    print("=== TEST 2: Semantic Advantages on Complex Sentences ===")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_cases = create_complex_test_cases()
    semantic_learning = SemanticAwareLearning(384)
    
    results_summary = []
    
    for case in test_cases:
        print(f"\n--- {case['category'].upper()} ---")
        print(f"Sentence: {case['sentence']}")
        print(f"Challenge: {case['challenge']}")
        
        words = case['sentence'].split()
        embeddings = model.encode(words)
        
        # Naive splines baseline
        t_original = np.linspace(0, 1, len(words))
        spline_funcs = []
        for dim in range(embeddings.shape[1]):
            spline_funcs.append(interp1d(t_original, embeddings[:, dim], kind='cubic'))
        
        def sample_naive_spline(t_vals):
            result = np.zeros((len(t_vals), embeddings.shape[1]))
            for dim, func in enumerate(spline_funcs):
                result[:, dim] = func(t_vals)
            return result
        
        # Sample at higher resolution for trajectory quality assessment
        t_dense = np.linspace(0, 1, 50)
        naive_trajectory = sample_naive_spline(t_dense)
        
        # Enhanced learnable splines with semantic awareness
        learnable_spline = LearnableSplineTrajectory(embeddings.shape[1])
        
        # Custom training with semantic losses
        embeddings_tensor = torch.FloatTensor(embeddings)
        learnable_spline.model = LearnableSplines(len(words), embeddings.shape[1])
        optimizer = torch.optim.Adam(learnable_spline.model.parameters(), lr=0.01)
        
        print("Training with semantic awareness...")
        for epoch in range(300):
            optimizer.zero_grad()
            
            # Standard reconstruction loss
            t_train = torch.linspace(0, 1, len(words))
            pred_trajectory = learnable_spline.model(embeddings_tensor, t_train)
            reconstruction_loss = nn.MSELoss()(pred_trajectory, embeddings_tensor)
            
            # Semantic losses
            semantic_tension_loss = semantic_learning.semantic_tension_loss(learnable_spline.model, embeddings_tensor)
            density_reg = semantic_learning.embedding_density_regularization(learnable_spline.model, embeddings_tensor)
            
            # Combined loss
            total_loss = reconstruction_loss + 0.1 * semantic_tension_loss + 0.05 * density_reg
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 75 == 0:
                print(f"  Epoch {epoch:3d}: Total={total_loss.item():.6f}, Recon={reconstruction_loss.item():.6f}, Sem={semantic_tension_loss.item():.6f}")
        
        # Sample learnable trajectory
        learnable_trajectory = learnable_spline.sample_trajectory(embeddings, t_dense)
        
        # Evaluation metrics
        
        # 1. Trajectory smoothness (second derivative magnitude)
        def compute_smoothness(trajectory):
            if len(trajectory) < 3:
                return 0
            second_deriv = np.diff(trajectory, n=2, axis=0)
            return np.mean(np.linalg.norm(second_deriv, axis=1))
        
        naive_smoothness = compute_smoothness(naive_trajectory)
        learnable_smoothness = compute_smoothness(learnable_trajectory)
        
        # 2. Semantic coherence (how well trajectory respects semantic similarities)
        def semantic_coherence_score(trajectory, original_embeddings):
            # Sample points along trajectory
            n_samples = min(20, len(trajectory))
            sample_indices = np.linspace(0, len(trajectory)-1, n_samples).astype(int)
            sampled_points = trajectory[sample_indices]
            
            # For each sampled point, find distance to nearest original word
            coherence_scores = []
            for point in sampled_points:
                distances = [np.linalg.norm(point - emb) for emb in original_embeddings]
                min_distance = min(distances)
                coherence_scores.append(1.0 / (1.0 + min_distance))  # Higher is better
            
            return np.mean(coherence_scores)
        
        naive_coherence = semantic_coherence_score(naive_trajectory, embeddings)
        learnable_coherence = semantic_coherence_score(learnable_trajectory, embeddings)
        
        # 3. Reconstruction accuracy at word points (SHOULD be 0 for both)
        naive_recon = sample_naive_spline(t_original)
        learnable_recon = learnable_spline.sample_trajectory(embeddings, t_original)
        
        naive_word_error = max([np.linalg.norm(naive_recon[i] - embeddings[i]) for i in range(len(words))])
        learnable_word_error = max([np.linalg.norm(learnable_recon[i] - embeddings[i]) for i in range(len(words))])
        
        # 4. MORE MEANINGFUL TEST: Reconstruction at OFFSET points (between words)
        # Sample at midpoints between words to test true interpolation quality
        t_offset = t_original[:-1] + 0.1 * (t_original[1] - t_original[0])  # Slightly offset from word positions
        
        naive_offset = sample_naive_spline(t_offset)
        learnable_offset = learnable_spline.sample_trajectory(embeddings, t_offset)
        
        # Compare interpolation quality by finding nearest word embeddings
        def interpolation_error(trajectory_points, original_embeddings):
            errors = []
            for point in trajectory_points:
                # Distance to nearest original word embedding
                distances = [np.linalg.norm(point - emb) for emb in original_embeddings]
                min_distance = min(distances)
                errors.append(min_distance)
            return np.mean(errors)
        
        naive_interp_error = interpolation_error(naive_offset, embeddings)
        learnable_interp_error = interpolation_error(learnable_offset, embeddings)
        
        # 5. INTUITIVE TEST: Word reconstruction quality
        # Sample densely and quantize to see actual word flow
        t_reconstruction = np.linspace(0, 1, 15)  # Dense sampling for word flow
        naive_flow = sample_naive_spline(t_reconstruction)
        learnable_flow = learnable_spline.sample_trajectory(embeddings, t_reconstruction)
        
        # REALITY CHECK: Use large vocabulary like main.py for realistic results
        try:
            nltk.data.find('corpora/brown')
        except LookupError:
            print("  Downloading NLTK brown corpus...")
            nltk.download('brown')
        
        # Get ~5000 most common words from Brown corpus (same as main.py)
        word_freq = nltk.FreqDist(brown.words())
        large_vocab = [word.lower() for word, freq in word_freq.most_common(5000) 
                       if word.isalpha() and len(word) > 1]
        
        print(f"  Using LARGE vocabulary: {len(large_vocab)} words (realistic test)")
        vocab_embeddings = model.encode(large_vocab)
        
        def quantize_flow(flow_embeddings):
            quantized = []
            for emb in flow_embeddings:
                distances = [np.linalg.norm(emb - vocab_emb) for vocab_emb in vocab_embeddings]
                closest_idx = np.argmin(distances)
                quantized.append(large_vocab[closest_idx])
            return quantized
        
        naive_word_flow = quantize_flow(naive_flow)
        learnable_word_flow = quantize_flow(learnable_flow)
        
        # 6. REALISTIC ACCURACY: Test word-level reconstruction with large vocab
        # Sample at original word positions and check accuracy (like main.py)
        def quantize_to_words(embeddings_array, vocabulary):
            vocab_embeddings = model.encode(vocabulary)
            quantized = []
            for emb in embeddings_array:
                # Cosine similarity (same as main.py)
                similarities = np.dot(vocab_embeddings, emb) / (
                    np.linalg.norm(vocab_embeddings, axis=1) * np.linalg.norm(emb)
                )
                best_idx = np.argmax(similarities)
                quantized.append(vocabulary[best_idx])
            return quantized
        
        naive_recon_large = quantize_to_words(naive_recon, large_vocab)
        learnable_recon_large = quantize_to_words(learnable_recon, large_vocab)
        
        def calculate_accuracy(predicted_words, original_words):
            if len(predicted_words) != len(original_words):
                return 0.0
            matches = sum(1 for p, o in zip(predicted_words, original_words) if p.lower() == o.lower())
            return matches / len(original_words)
        
        naive_accuracy = calculate_accuracy(naive_recon_large, words)
        learnable_accuracy = calculate_accuracy(learnable_recon_large, words)
        
        # Results
        print(f"Results:")
        print(f"  Smoothness:     Naive={naive_smoothness:.4f}, Learnable={learnable_smoothness:.4f} ({(learnable_smoothness/naive_smoothness-1)*100:+.1f}%)")
        print(f"  Coherence:      Naive={naive_coherence:.4f}, Learnable={learnable_coherence:.4f} ({(learnable_coherence/naive_coherence-1)*100:+.1f}%)")
        print(f"  Word Error:     Naive={naive_word_error:.6f}, Learnable={learnable_word_error:.6f} (Expected: both ~0)")
        print(f"  Interp Error:   Naive={naive_interp_error:.4f}, Learnable={learnable_interp_error:.4f} ({(learnable_interp_error/naive_interp_error-1)*100:+.1f}%)")
        print(f"  REALITY CHECK - Large Vocab Accuracy:")
        print(f"    Naive:     {naive_accuracy:.1%} ({sum(1 for p, o in zip(naive_recon_large, words) if p.lower() == o.lower())}/{len(words)} words correct)")
        print(f"    Learnable: {learnable_accuracy:.1%} ({sum(1 for p, o in zip(learnable_recon_large, words) if p.lower() == o.lower())}/{len(words)} words correct)")
        
        # Show actual reconstructed sentences for reality check
        print(f"\n  Large Vocabulary Reconstruction:")
        print(f"  Original:  {' '.join(words)}")
        print(f"  Naive:     {' '.join(naive_recon_large)}")
        print(f"  Learnable: {' '.join(learnable_recon_large)}")
        
        # Show word flows for intuitive understanding (this might be messy with large vocab)
        print(f"\n  Word Flow Comparison (15 samples, large vocab):")
        print(f"  Original:  {' → '.join(words)}")
        print(f"  Naive:     {' → '.join(naive_word_flow[:8])} ...")  # Show first 8 to avoid clutter
        print(f"  Learnable: {' → '.join(learnable_word_flow[:8])} ...")
        
        # Show learned parameters
        params = learnable_spline.get_learned_parameters()
        print(f"\n  Learned tensions: {params['tension_params']}")
        
        # Determine if learnable is better (updated for realistic large vocab testing)
        improvements = 0
        if learnable_coherence > naive_coherence * 1.02:  # 2% improvement threshold
            improvements += 1
            print(f"  ✅ Better semantic coherence")
        if learnable_smoothness < naive_smoothness * 0.98:  # 2% improvement threshold  
            improvements += 1
            print(f"  ✅ Smoother trajectory")
        if learnable_interp_error < naive_interp_error * 0.98:  # Better interpolation between words
            improvements += 1
            print(f"  ✅ Better interpolation quality")
        if learnable_accuracy >= naive_accuracy:  # REALISTIC: Large vocab accuracy comparison
            improvements += 1
            print(f"  ✅ Better/equal large vocab accuracy ({learnable_accuracy:.1%} vs {naive_accuracy:.1%})")
        else:
            print(f"  ⚠️ Lower large vocab accuracy ({learnable_accuracy:.1%} vs {naive_accuracy:.1%})")
        
        # BONUS: Check if learnable maintains reasonable accuracy despite complexity  
        if learnable_accuracy >= 0.5:  # At least 50% accuracy with large vocab
            improvements += 0.5
            print(f"  ✅ Maintains reasonable accuracy (≥50%) with large vocabulary")
        elif learnable_accuracy >= 0.3:  # At least 30% accuracy
            print(f"  ⚠️ Moderate accuracy (≥30%) with large vocabulary")
        else:
            print(f"  ❌ Low accuracy (<30%) with large vocabulary")
        
        success = improvements >= 2
        status = "🎉 BETTER" if success else "⚠️  SIMILAR"
        print(f"  Overall: {status}")
        
        results_summary.append({
            'category': case['category'],
            'sentence': case['sentence'],
            'naive_smoothness': naive_smoothness,
            'learnable_smoothness': learnable_smoothness,
            'naive_coherence': naive_coherence,
            'learnable_coherence': learnable_coherence,
            'naive_interp_error': naive_interp_error,
            'learnable_interp_error': learnable_interp_error,
            'naive_accuracy': naive_accuracy,  # NEW: Large vocab accuracy
            'learnable_accuracy': learnable_accuracy,  # NEW: Large vocab accuracy
            'naive_recon_large': naive_recon_large,  # NEW: Actual reconstructions
            'learnable_recon_large': learnable_recon_large,  # NEW: Actual reconstructions
            'naive_word_flow': naive_word_flow,
            'learnable_word_flow': learnable_word_flow,
            'success': success,
            'improvements': improvements,
            'learned_tensions': params['tension_params']
        })
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    successful_cases = sum(1 for r in results_summary if r['success'])
    total_cases = len(results_summary)
    
    print(f"Successful cases: {successful_cases}/{total_cases} ({successful_cases/total_cases*100:.1f}%)")
    
    # REALITY CHECK SUMMARY: Large vocabulary performance
    avg_naive_accuracy = np.mean([r['naive_accuracy'] for r in results_summary])
    avg_learnable_accuracy = np.mean([r['learnable_accuracy'] for r in results_summary])
    accuracy_improvement = avg_learnable_accuracy - avg_naive_accuracy
    
    print(f"\nREALITY CHECK - Large Vocabulary Performance:")
    print(f"  Average accuracy (5000-word vocabulary):")
    print(f"    Naive splines:     {avg_naive_accuracy:.1%}")
    print(f"    Learnable splines: {avg_learnable_accuracy:.1%}")
    print(f"    Improvement:       {accuracy_improvement:+.1%}")
    
    # Show range of accuracies
    naive_accuracies = [r['naive_accuracy'] for r in results_summary]
    learnable_accuracies = [r['learnable_accuracy'] for r in results_summary]
    
    print(f"  Accuracy ranges:")
    print(f"    Naive:     {min(naive_accuracies):.1%} - {max(naive_accuracies):.1%}")
    print(f"    Learnable: {min(learnable_accuracies):.1%} - {max(learnable_accuracies):.1%}")
    
    if successful_cases >= total_cases * 0.67:  # 67% success threshold
        print(f"\n🎉 SUCCESS! Learnable splines show semantic advantages on complex sentences!")
        print(f"Key improvements:")
        for result in results_summary:
            if result['success']:
                naive_acc = result['naive_accuracy']
                learn_acc = result['learnable_accuracy']
                print(f"  ✅ {result['category']}: {result['improvements']:.1f} metrics improved, accuracy {naive_acc:.1%}→{learn_acc:.1%}")
    else:
        print(f"\n⚠️ Mixed results. Need further refinement.")
        print(f"Areas for improvement:")
        for result in results_summary:
            if not result['success']:
                print(f"  - {result['category']}: Only {result['improvements']:.1f} metrics improved")
    
    # Create comparison visualizations
    create_comparison_charts(results_summary)
    
    return results_summary


def create_comparison_charts(results_summary):
    """Create visual comparisons of the results"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create comprehensive comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))  # 2x4 grid to include accuracy
    fig.suptitle('Learnable Splines vs Naive Splines: Comprehensive Comparison (Large Vocabulary)', fontsize=16, fontweight='bold')
    
    categories = [r['category'] for r in results_summary]
    
    # Plot 1: Smoothness Comparison
    ax1 = axes[0, 0]
    naive_smooth = [r['naive_smoothness'] for r in results_summary]
    learnable_smooth = [r['learnable_smoothness'] for r in results_summary]
    
    x = range(len(categories))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], naive_smooth, width, label='Naive Splines', alpha=0.8)
    ax1.bar([i + width/2 for i in x], learnable_smooth, width, label='Learnable Splines', alpha=0.8)
    ax1.set_ylabel('Trajectory Roughness')
    ax1.set_title('Smoothness Comparison\n(Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(' ', '\n') for c in categories], rotation=0, ha='center')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Semantic Coherence
    ax2 = axes[0, 1]
    naive_coherence = [r['naive_coherence'] for r in results_summary]
    learnable_coherence = [r['learnable_coherence'] for r in results_summary]
    
    ax2.bar([i - width/2 for i in x], naive_coherence, width, label='Naive Splines', alpha=0.8)
    ax2.bar([i + width/2 for i in x], learnable_coherence, width, label='Learnable Splines', alpha=0.8)
    ax2.set_ylabel('Semantic Coherence Score')
    ax2.set_title('Semantic Coherence\n(Higher is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace(' ', '\n') for c in categories], rotation=0, ha='center')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Interpolation Error
    ax3 = axes[0, 2]
    naive_interp = [r['naive_interp_error'] for r in results_summary]
    learnable_interp = [r['learnable_interp_error'] for r in results_summary]
    
    ax3.bar([i - width/2 for i in x], naive_interp, width, label='Naive Splines', alpha=0.8)
    ax3.bar([i + width/2 for i in x], learnable_interp, width, label='Learnable Splines', alpha=0.8)
    ax3.set_ylabel('Interpolation Error')
    ax3.set_title('Interpolation Quality\n(Lower is Better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace(' ', '\n') for c in categories], rotation=0, ha='center')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Large Vocabulary Accuracy (NEW!)
    ax4 = axes[0, 3]
    naive_accuracy = [r['naive_accuracy'] for r in results_summary]
    learnable_accuracy = [r['learnable_accuracy'] for r in results_summary]
    
    ax4.bar([i - width/2 for i in x], naive_accuracy, width, label='Naive Splines', alpha=0.8)
    ax4.bar([i + width/2 for i in x], learnable_accuracy, width, label='Learnable Splines', alpha=0.8)
    ax4.set_ylabel('Accuracy (5000-word vocab)')
    ax4.set_title('Large Vocabulary Accuracy\n(Higher is Better)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([c.replace(' ', '\n') for c in categories], rotation=0, ha='center')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.0)  # 0-100% range
    
    # Plot 5: Improvement Percentages
    ax5 = axes[1, 0]
    smoothness_improvement = [(r['naive_smoothness'] - r['learnable_smoothness']) / r['naive_smoothness'] * 100 
                             for r in results_summary]
    coherence_improvement = [(r['learnable_coherence'] - r['naive_coherence']) / r['naive_coherence'] * 100 
                            for r in results_summary]
    interp_improvement = [(r['naive_interp_error'] - r['learnable_interp_error']) / r['naive_interp_error'] * 100 
                         for r in results_summary]
    accuracy_improvement = [(r['learnable_accuracy'] - r['naive_accuracy']) * 100  # Already in percentage form
                           for r in results_summary]
    
    ax5.bar([i - width/4 for i in x], smoothness_improvement, width/2, label='Smoothness', alpha=0.8)
    ax5.bar([i - width/8 for i in x], coherence_improvement, width/2, label='Coherence', alpha=0.8)
    ax5.bar([i + width/8 for i in x], interp_improvement, width/2, label='Interpolation', alpha=0.8)
    ax5.bar([i + width/4 for i in x], accuracy_improvement, width/2, label='Accuracy', alpha=0.8)
    ax5.set_ylabel('Improvement (%)')
    ax5.set_title('Percentage Improvements\n(Higher is Better)')
    ax5.set_xticks(x)
    ax5.set_xticklabels([c.replace(' ', '\n') for c in categories], rotation=0, ha='center')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 6: Learned Tension Parameters
    ax6 = axes[1, 1]
    tension_data = []
    tension_labels = []
    for i, result in enumerate(results_summary):
        tensions = result['learned_tensions']
        for j, tension in enumerate(tensions):
            tension_data.append(tension)
            tension_labels.append(f"{result['category']}\nSegment {j+1}")
    
    # Show distribution of learned tensions
    ax6.hist([r['learned_tensions'].flatten() for r in results_summary], 
             bins=20, alpha=0.7, label=[r['category'] for r in results_summary])
    ax6.set_xlabel('Learned Tension Value')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Learned\nTension Parameters')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Success Summary
    ax7 = axes[1, 2]
    success_counts = [r['improvements'] for r in results_summary]
    colors = ['#2ecc71' if s >= 3 else '#f39c12' if s >= 2 else '#e74c3c' for s in success_counts]
    
    bars = ax7.bar(range(len(categories)), success_counts, color=colors, alpha=0.8)
    ax7.set_ylabel('Number of Metrics Improved')
    ax7.set_title('Success Summary\n(Out of 4+ Metrics)')
    ax7.set_xticks(range(len(categories)))
    ax7.set_xticklabels([c.replace(' ', '\n') for c in categories], rotation=0, ha='center')
    ax7.set_ylim(0, 5)  # Increased because we now have 4+ metrics
    ax7.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, success_counts)):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 8: Accuracy Range Comparison (use the last slot)
    ax8 = axes[1, 3]
    naive_accuracies = [r['naive_accuracy'] for r in results_summary]
    learnable_accuracies = [r['learnable_accuracy'] for r in results_summary]
    
    # Box plot showing accuracy distributions
    ax8.boxplot([naive_accuracies, learnable_accuracies], 
                labels=['Naive', 'Learnable'], patch_artist=True)
    ax8.set_ylabel('Accuracy')
    ax8.set_title('Accuracy Distribution\n(Large Vocabulary)')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('learnable_splines_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved comprehensive comparison chart: learnable_splines_comparison.png")
    plt.show()
    
    # Create word flow visualization
    create_word_flow_chart(results_summary)


def create_word_flow_chart(results_summary):
    """Create visualization of word flows for intuitive understanding"""
    
    fig, axes = plt.subplots(len(results_summary), 1, figsize=(16, 3 * len(results_summary)))
    if len(results_summary) == 1:
        axes = [axes]
    
    fig.suptitle('Word Flow Comparison: Naive vs Learnable Splines', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(results_summary):
        ax = axes[i]
        
        # Get original words and REALISTIC flows (large vocabulary reconstructions)
        original_words = result['sentence'].split()
        naive_recon = result['naive_recon_large']  # Realistic reconstruction
        learnable_recon = result['learnable_recon_large']  # Realistic reconstruction
        
        # Create flow comparison
        y_positions = [2, 1, 0]
        flow_labels = ['Original', 'Naive Spline (5K vocab)', 'Learnable Spline (5K vocab)']
        flows = [original_words, naive_recon, learnable_recon]
        colors = ['black', 'blue', 'red']
        
        for j, (flow, label, color) in enumerate(zip(flows, flow_labels, colors)):
            y = y_positions[j]
            
            # Plot words as points
            x_positions = np.linspace(0, 10, len(flow))
            ax.scatter(x_positions, [y] * len(flow), c=color, s=100, alpha=0.7, zorder=3)
            
            # Add word labels
            for k, (x, word) in enumerate(zip(x_positions, flow)):
                ax.annotate(word, (x, y), xytext=(0, 10), textcoords='offset points', 
                           ha='center', va='bottom', fontsize=8, rotation=0)
            
            # Draw flow line
            if j > 0:  # Don't draw line for original (it's just the reference)
                ax.plot(x_positions, [y] * len(flow), color=color, alpha=0.5, linewidth=2, zorder=1)
        
        # Formatting
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlim(-0.5, 10.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(flow_labels)
        ax.set_xlabel('Trajectory Progress →')
        ax.set_title(f'{result["category"]}: "{result["sentence"]}"', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add improvement metrics as text
        interp_improvement = (result['naive_interp_error'] - result['learnable_interp_error']) / result['naive_interp_error'] * 100
        accuracy_improvement = (result['learnable_accuracy'] - result['naive_accuracy']) * 100
        
        ax.text(0.02, 0.95, f'Accuracy: {result["naive_accuracy"]:.1%} → {result["learnable_accuracy"]:.1%} ({accuracy_improvement:+.1f}%)', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.02, 0.85, f'Interpolation: {interp_improvement:+.1f}% better', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('word_flow_comparison.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved word flow comparison: word_flow_comparison.png")
    plt.show()


if __name__ == "__main__":
    # Step 1: Verify we can match naive splines  
    print("🔧 STEP 1: Basic Performance Verification")
    basic_results = test_basic_performance()
    
    if basic_results['overall_success']:
        print(f"\n🚀 STEP 2: Testing Semantic Advantages")
        semantic_results = test_semantic_advantages()
    else:
        print(f"\n⚠️ Basic performance issues - skipping complex tests")