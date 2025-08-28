#!/usr/bin/env python3
"""
TextAttack implementation against Google Cloud NLP using TextBugger attack.
Tests with 100 examples from Rotten Tomatoes dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import textattack
    from textattack.models.wrappers import ModelWrapper
    from textattack.datasets import Dataset
    from textattack.attack_recipes import TextBuggerLi2018
    from textattack import AttackArgs, Attacker
except ImportError:
    print("TextAttack not found. Installing...")
    os.system("pip install textattack")
    import textattack
    from textattack.models.wrappers import ModelWrapper
    from textattack.datasets import Dataset
    from textattack.attack_recipes import TextBuggerLi2018
    from textattack import AttackArgs, Attacker

from clients.google_nlp import GoogleNLPClassifier

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/textattack_google_nlp_{timestamp}.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GoogleNLPWrapper(ModelWrapper):
    """TextAttack wrapper for Google Cloud NLP sentiment analysis."""
    
    def __init__(self, credentials_path=None):
        self.model = GoogleNLPClassifier(credentials_path)
        # Map labels to indices for TextAttack compatibility
        self.label_map = {"negative": 0, "positive": 1, "neutral": 2}
        self.reverse_label_map = {0: "negative", 1: "positive", 2: "neutral"}
        
    def __call__(self, text_input_list: List[str]) -> np.ndarray:
        """
        Predict sentiment for a list of texts.
        
        Args:
            text_input_list: List of text strings to classify
            
        Returns:
            numpy array of prediction probabilities with shape (batch_size, num_classes)
        """
        predictions = []
        
        for text in text_input_list:
            try:
                label, scores = self.model.classify(text)
                
                # Convert to probability distribution over [negative, positive, neutral]
                prob_vec = [
                    scores.get("negative", 0.0),
                    scores.get("positive", 0.0), 
                    scores.get("neutral", 0.0)
                ]
                
                # Normalize to ensure probabilities sum to 1
                total = sum(prob_vec)
                if total > 0:
                    prob_vec = [p / total for p in prob_vec]
                else:
                    # Default uniform distribution if no valid scores
                    prob_vec = [0.33, 0.33, 0.34]
                    
                predictions.append(prob_vec)
                
            except Exception as e:
                logger.warning(f"Error classifying text: {e}")
                # Default to neutral prediction on error
                predictions.append([0.33, 0.33, 0.34])
                
        return np.array(predictions)


def load_rtmr_dataset(num_samples: int = 100) -> Dataset:
    """Load Rotten Tomatoes dataset for TextAttack."""
    
    # Load test data
    X_test_path = "datasets/rtmr/X_test.csv"
    y_test_path = "datasets/rtmr/y_test.csv"
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Dataset files not found. Expected {X_test_path} and {y_test_path}")
    
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    
    # Take first num_samples examples
    texts = X_test['text'].iloc[:num_samples].tolist()
    labels = y_test['label'].iloc[:num_samples].tolist()
    
    logger.info(f"Loaded {len(texts)} examples from RTMR dataset")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Create TextAttack dataset
    dataset = Dataset(list(zip(texts, labels)))
    return dataset


def run_textbugger_attack(num_samples: int = 100, credentials_path: str = None):
    """Run TextBugger attack against Google Cloud NLP."""
    
    logger.info("Starting TextBugger attack against Google Cloud NLP")
    logger.info(f"Testing with {num_samples} samples from Rotten Tomatoes dataset")
    
    try:
        # Load dataset
        dataset = load_rtmr_dataset(num_samples)
        
        # Create model wrapper
        model_wrapper = GoogleNLPWrapper(credentials_path)
        
        # Build TextBugger attack
        attack = TextBuggerLi2018.build(model_wrapper)
        
        # Configure attack arguments
        attack_args = AttackArgs(
            num_examples=num_samples,
            log_to_csv=f"results/rtmr/textattack_google_nlp_{timestamp}.csv",
            csv_coloring_style="plain",
            num_examples_offset=0,
            attack_n=True,
            shuffle=False
        )
        
        # Create attacker
        attacker = Attacker(attack, dataset, attack_args)
        
        # Run attack
        logger.info("Running TextBugger attack...")
        attack_results = attacker.attack_dataset()
        
        # Log results summary
        successful_attacks = sum(1 for result in attack_results if result.succeeded)
        failed_attacks = sum(1 for result in attack_results if result.failed)
        skipped_attacks = sum(1 for result in attack_results if result.skipped)
        
        logger.info(f"Attack Results Summary:")
        logger.info(f"  Total examples: {len(attack_results)}")
        logger.info(f"  Successful attacks: {successful_attacks}")
        logger.info(f"  Failed attacks: {failed_attacks}")
        logger.info(f"  Skipped attacks: {skipped_attacks}")
        logger.info(f"  Success rate: {successful_attacks/len(attack_results)*100:.2f}%")
        
        # Save detailed results
        results_data = []
        for i, result in enumerate(attack_results):
            result_dict = {
                'example_id': i,
                'original_text': result.original_text(),
                'original_label': result.ground_truth_output,
                'original_prediction': result.original_result.output,
                'attack_success': result.succeeded,
                'perturbed_text': result.perturbed_text() if result.succeeded else None,
                'perturbed_prediction': result.perturbed_result.output if result.succeeded else None,
                'num_queries': result.num_queries,
                'attack_log': str(result)
            }
            results_data.append(result_dict)
        
        # Save to CSV
        results_df = pd.DataFrame(results_data)
        results_path = f"results/rtmr/textattack_detailed_google_nlp_{timestamp}.csv"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Detailed results saved to: {results_path}")
        logger.info(f"TextAttack CSV results saved to: results/rtmr/textattack_google_nlp_{timestamp}.csv")
        
        return attack_results
        
    except Exception as e:
        logger.error(f"Error during attack: {e}")
        raise


def main():
    """Main function to run the TextBugger attack."""
    
    # Check for Google Cloud credentials
    credentials_path = None
    if os.path.exists("credentials/textbugger3-c34a9b2b7c19.json"):
        credentials_path = "credentials/textbugger3-c34a9b2b7c19.json"
    elif os.path.exists("credentials/textbugger-test-2-87b3c4ef632f.json"):
        credentials_path = "credentials/textbugger-test-2-87b3c4ef632f.json"
    
    if not credentials_path and not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        logger.error("No Google Cloud credentials found. Please set GOOGLE_APPLICATION_CREDENTIALS or place credentials in credentials/ folder.")
        return
    
    try:
        # Run attack with 100 samples
        results = run_textbugger_attack(num_samples=100, credentials_path=credentials_path)
        logger.info("TextBugger attack completed successfully!")
        
    except Exception as e:
        logger.error(f"Attack failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()