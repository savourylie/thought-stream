import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.interpolate import interp1d
from typing import List, Tuple
import nltk
from nltk.corpus import brown
import random


class ThoughtTrajectory:
    """Core class for modeling text as continuous thought trajectories."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with a sentence transformer model."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def encode_sentence(self, sentence: str) -> np.ndarray:
        """Convert sentence to sequence of word embeddings."""
        words = sentence.split()
        embeddings = self.model.encode(words)
        return embeddings
    
    def fit_trajectory(self, embeddings: np.ndarray) -> callable:
        """Fit a smooth trajectory through word embeddings."""
        num_points = len(embeddings)
        t = np.linspace(0, 1, num_points)  # Time parameter
        
        # Create interpolation functions for each dimension
        trajectory_funcs = []
        for dim in range(embeddings.shape[1]):
            func = interp1d(t, embeddings[:, dim], kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
            trajectory_funcs.append(func)
        
        def trajectory(t_values):
            """Sample trajectory at given time points."""
            result = np.zeros((len(t_values), embeddings.shape[1]))
            for dim, func in enumerate(trajectory_funcs):
                result[:, dim] = func(t_values)
            return result
        
        return trajectory
    
    def quantize_to_words(self, embeddings: np.ndarray, vocabulary: List[str]) -> List[str]:
        """Find nearest words in vocabulary for given embeddings."""
        vocab_embeddings = self.model.encode(vocabulary)
        
        words = []
        for emb in embeddings:
            # Find closest word by cosine similarity
            similarities = np.dot(vocab_embeddings, emb) / (
                np.linalg.norm(vocab_embeddings, axis=1) * np.linalg.norm(emb)
            )
            best_idx = np.argmax(similarities)
            words.append(vocabulary[best_idx])
        
        return words
    
    def reconstruct_sentence(self, trajectory: callable, num_samples: int, 
                           vocabulary: List[str]) -> str:
        """Reconstruct sentence by sampling trajectory and quantizing."""
        t_values = np.linspace(0, 1, num_samples)
        sampled_embeddings = trajectory(t_values)
        words = self.quantize_to_words(sampled_embeddings, vocabulary)
        return ' '.join(words)
    
    def interpolate_sentences(self, sentence1: str, sentence2: str, 
                            num_steps: int = 5) -> List[str]:
        """Create interpolated sentences between two input sentences."""
        # Get trajectories for both sentences
        emb1 = self.encode_sentence(sentence1)
        emb2 = self.encode_sentence(sentence2)
        
        traj1 = self.fit_trajectory(emb1)
        traj2 = self.fit_trajectory(emb2)
        
        # Create combined vocabulary
        words1 = sentence1.split()
        words2 = sentence2.split()
        vocabulary = list(set(words1 + words2 + ["the", "and", "to", "a", "is", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "into", "him", "has", "two", "more", "very", "what", "know", "just", "first", "get", "over", "think", "also", "your", "work", "life", "only", "can", "still", "should", "after", "being", "now", "made", "before", "here", "through", "when", "where", "much", "go", "me", "world", "too", "any", "may", "say", "most", "such"]))
        
        interpolated = []
        alpha_values = np.linspace(0, 1, num_steps)
        
        for alpha in alpha_values:
            # Create interpolated trajectory
            def interpolated_trajectory(t_values):
                points1 = traj1(t_values)
                points2 = traj2(t_values)
                return (1 - alpha) * points1 + alpha * points2
            
            # Sample interpolated trajectory
            avg_length = (len(words1) + len(words2)) // 2
            interpolated_sentence = self.reconstruct_sentence(
                interpolated_trajectory, avg_length, vocabulary
            )
            interpolated.append(interpolated_sentence)
        
        return interpolated


def demo_basic_reconstruction():
    """Demonstrate basic trajectory fitting and reconstruction."""
    print("\n=== ThoughtStream Phase 1 Demo ===")
    
    # Initialize system
    ts = ThoughtTrajectory()
    
    # Test sentence
    sentence = "The cat sat on the mat"
    print(f"\nOriginal: {sentence}")
    
    # Get embeddings for each word
    embeddings = ts.encode_sentence(sentence)
    print(f"Encoded {len(embeddings)} words to {embeddings.shape[1]}D embeddings")
    
    # Fit trajectory
    trajectory = ts.fit_trajectory(embeddings)
    print("Fitted smooth trajectory through word embeddings")
    
    # Simple vocabulary (in practice, this would be much larger)
    vocabulary = sentence.split() + ["dog", "mouse", "chair", "table", "floor"]
    
    # Reconstruct with same number of samples
    reconstructed = ts.reconstruct_sentence(trajectory, len(embeddings), vocabulary)
    print(f"Reconstructed: {reconstructed}")
    
    # Try with fewer samples (compression)
    compressed = ts.reconstruct_sentence(trajectory, 3, vocabulary)
    print(f"Compressed (3 words): {compressed}")
    
    return ts, trajectory


def demo_interpolation():
    """Demonstrate interpolation between two sentences."""
    print("\n=== Sentence Interpolation Demo ===")
    
    ts = ThoughtTrajectory()
    
    # Two different sentences
    sentence1 = "The cat sits quietly"
    sentence2 = "The dog runs quickly"
    
    print(f"Sentence 1: {sentence1}")
    print(f"Sentence 2: {sentence2}")
    print("\nInterpolation:")
    
    interpolated = ts.interpolate_sentences(sentence1, sentence2, num_steps=7)
    for i, sentence in enumerate(interpolated):
        alpha = i / (len(interpolated) - 1)
        print(f"  {alpha:.2f}: {sentence}")
    
    return ts


def demo_large_vocabulary():
    """Demonstrate reconstruction with a large vocabulary (reality check)."""
    print("\n=== Large Vocabulary Reality Check ===")
    
    ts = ThoughtTrajectory()
    
    # Get a large vocabulary from NLTK
    try:
        # Try to download brown corpus if not already present
        nltk.data.find('corpora/brown')
    except LookupError:
        print("Downloading NLTK brown corpus...")
        nltk.download('brown')
    
    # Get ~5000 most common words from Brown corpus
    word_freq = nltk.FreqDist(brown.words())
    large_vocabulary = [word.lower() for word, freq in word_freq.most_common(5000) 
                       if word.isalpha() and len(word) > 1]
    
    print(f"Using vocabulary of {len(large_vocabulary)} words")
    print(f"Sample words: {large_vocabulary[:20]}")
    
    # Test sentences
    test_sentences = [
        "The cat sat on the mat",
        "I love programming with Python", 
        "The weather is beautiful today",
        "Machine learning transforms data into insights"
    ]
    
    for sentence in test_sentences:
        print(f"\n--- Testing: '{sentence}' ---")
        
        # Original small vocabulary (sentence words only)
        small_vocab = sentence.split() + ["a", "the", "is", "was", "and", "to"]
        
        # Encode and fit trajectory
        embeddings = ts.encode_sentence(sentence)
        trajectory = ts.fit_trajectory(embeddings)
        
        # Reconstruct with small vocabulary
        small_recon = ts.reconstruct_sentence(trajectory, len(embeddings), small_vocab)
        
        # Reconstruct with large vocabulary  
        large_recon = ts.reconstruct_sentence(trajectory, len(embeddings), large_vocabulary)
        
        print(f"Original:     {sentence}")
        print(f"Small vocab:  {small_recon}")
        print(f"Large vocab:  {large_recon}")
        
        # Show how much they differ
        original_words = sentence.split()
        large_words = large_recon.split()
        
        if len(original_words) == len(large_words):
            matches = sum(1 for o, l in zip(original_words, large_words) if o.lower() == l.lower())
            accuracy = matches / len(original_words)
            print(f"Accuracy:     {accuracy:.1%} ({matches}/{len(original_words)} words correct)")
        
    return ts


def main():
    try:
        print("🚀 ThoughtStream Phase 1: Proof of Concept")
        
        # Demo 1: Basic reconstruction
        ts, trajectory = demo_basic_reconstruction()
        
        # Demo 2: Interpolation
        demo_interpolation()
        
        # Demo 3: Large vocabulary reality check
        demo_large_vocabulary()
        
        print("\n✅ Phase 1 complete! All three goals achieved:")
        print("  ✓ Basic trajectory fitting through word embeddings")
        print("  ✓ Demonstrate reconstruction on simple sentences")
        print("  ✓ Show interpolation between two sentences")
        print("  ⚠️ Revealed limitations with large vocabularies")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
