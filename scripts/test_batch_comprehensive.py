#!/usr/bin/env python3
"""
Comprehensive Batch TextBugger Test

Tests batch attack methods on 100 RTMR examples with Google Cloud NLP.
Tracks detailed metrics including semantic similarity and perturbation ratios.

Usage:
    python scripts/test_batch_comprehensive.py --method multi_word --batch_size 2
    python scripts/test_batch_comprehensive.py --method confidence_driven
    python scripts/test_batch_comprehensive.py --method adaptive --batch_size 3
"""

import argparse
import os
import sys
import csv
import re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from attacks.batch_blackbox import BatchBlackBoxTextBugger, BatchLevel
from attacks.blackbox import SemanticSimilarity
from models.api_models import APIModelWrapper
from clients.ibm_watson import IBMWatsonClassifier
from clients.google_nlp import GoogleNLPClassifier
from utils.text_processing import split_sentences, split_words, fix_spacing

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging with timestamps"""
    import logging
    
    # Suppress verbose logging
    logging.getLogger('google').setLevel(logging.WARNING)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/batch_comprehensive_{timestamp}.log'
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Log file: {log_filename}")
    return logger, log_filename

def load_rtmr_dataset(limit=100):
    """Load RTMR dataset with specified limit"""
    dataset_path = Path("datasets/rtmr/X_test.csv")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"RTMR dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    texts = df["text"].head(limit).tolist()
    
    return texts

def calculate_string_similarity(text1: str, text2: str) -> float:
    """
    Calculate string similarity score using SequenceMatcher
    Returns the string similarity ratio (0.0-1.0) for consistency with attack similarity checks
    """
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_perturbation_ratio(original: str, adversarial: str) -> float:
    """Calculate the ratio of perturbed words"""
    original_words = split_words(original)
    adversarial_words = split_words(adversarial)
    
    # Handle different lengths (shouldn't happen with our perturbations, but just in case)
    min_length = min(len(original_words), len(adversarial_words))
    
    if min_length == 0:
        return 0.0
    
    # Count different words
    different_words = 0
    for i in range(min_length):
        if i < len(original_words) and i < len(adversarial_words):
            if original_words[i] != adversarial_words[i]:
                different_words += 1
    
    # Add any extra words as different
    different_words += abs(len(original_words) - len(adversarial_words))
    
    # Calculate ratio
    total_words = len(original_words)
    return different_words / total_words if total_words > 0 else 0.0

def save_results(results, method_name, batch_size, dataset_name="rtmr"):
    """Save comprehensive attack results to CSV"""
    results_dir = Path(f"results/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"batch_{method_name}_bs{batch_size}_comprehensive_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Enhanced header with new metrics
        writer.writerow([
            "example_id",
            "original_text", 
            "adversarial_text", 
            "original_label",
            "new_label",
            "requests_used", 
            "num_perturbations", 
            "attack_successful",
            "string_similarity",
            "perturbation_ratio",
            "method_name",
            "batch_size",
            "timestamp"
        ])
        
        for result in results:
            writer.writerow(result + [datetime.now().isoformat()])
    
    return filename

def run_batch_attack_comprehensive(method_name, batch_size, limit=100):
    """Run comprehensive batch attack with detailed metrics"""
    
    logger, log_filename = setup_logging()
    
    # Convert method name to BatchLevel
    method_map = {
        "single_word": BatchLevel.SINGLE_WORD,
        "multi_word": BatchLevel.MULTI_WORD,
        "adaptive_gradual": BatchLevel.ADAPTIVE_GRADUAL, 
        "adaptive_dynamic": BatchLevel.ADAPTIVE_DYNAMIC
    }
    
    if method_name not in method_map:
        raise ValueError(f"Unknown method: {method_name}. Choose from: {list(method_map.keys())}")
    
    batch_level = method_map[method_name]
    
    # Initialize Google Cloud NLP
    logger.info("🚀 Initializing Google Cloud NLP...")
    try:
        google_client = GoogleNLPClassifier()
        wrapper = APIModelWrapper(google_client)
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Cloud NLP: {e}")
        logger.info("💡 Make sure GOOGLE_APPLICATION_CREDENTIALS is set in your environment")
        return
    
    # Initialize batch attacker
    attacker = BatchBlackBoxTextBugger(
        wrapper, 
        similarity_threshold=0.8,
        batch_level=batch_level,
        batch_size=batch_size
    )
    
    # Use the same SemanticSimilarity instance for consistency
    semantic_similarity = attacker.similarity
    
    # Load dataset
    logger.info(f"📊 Loading {limit} examples from RTMR dataset...")
    try:
        texts = load_rtmr_dataset(limit)
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return
    
    logger.info(f"✅ Loaded {len(texts)} examples")
    logger.info(f"🎯 Method: {method_name} (batch_size={batch_size})")
    logger.info("="*80)
    
    # Run attacks
    results = []
    successful = 0
    total_requests = 0
    total_perturbations = 0
    semantic_similarities = []
    perturbation_ratios = []
    
    for i, text in enumerate(texts, 1):
        clean_text = re.sub(r'\s+', ' ', text).strip()
        logger.info(f"🎯 Attack {i}/{len(texts)}: {clean_text[:50]}{'...' if len(clean_text) > 50 else ''}")
        
        try:
            # Run batch attack (stops when no more candidates or similarity violated)
            adv_text, orig_label, num_reqs, num_perturbs, success = attacker.attack(clean_text)
            
            # Calculate new metrics
            if success:
                # For successful attacks, we know the label flipped, but let's get the actual new label
                new_label, _ = wrapper(adv_text)
                
                # Calculate string similarity for consistency with attack similarity checks
                string_sim = calculate_string_similarity(clean_text, adv_text)
                semantic_similarities.append(string_sim)
                
                # Calculate perturbation ratio
                pert_ratio = calculate_perturbation_ratio(clean_text, adv_text)
                perturbation_ratios.append(pert_ratio)
                
                successful += 1
                logger.info(f"  ✅ SUCCESS: {orig_label} → {new_label}")
                logger.info(f"     📊 Queries: {num_reqs}, Perturbations: {num_perturbs}")
                logger.info(f"     🔍 String Similarity: {string_sim:.3f}")
                logger.info(f"     🔄 Perturbation Ratio: {pert_ratio:.3f}")
            else:
                new_label = orig_label  # No flip
                string_sim = 1.0  # Same text
                pert_ratio = 0.0    # No perturbations in final result
                logger.info(f"  ❌ FAILED: {orig_label} (Queries: {num_reqs}, Perturbations: {num_perturbs})")
            
            total_requests += num_reqs
            total_perturbations += num_perturbs
            
            # Store comprehensive result
            results.append([
                i,                      # example_id
                clean_text,             # original_text
                adv_text,               # adversarial_text
                orig_label,             # original_label
                new_label,              # new_label
                num_reqs,               # requests_used
                num_perturbs,           # num_perturbations
                success,                # attack_successful
                string_sim,             # string_similarity
                pert_ratio,             # perturbation_ratio
                method_name,            # method_name
                batch_size              # batch_size
            ])
            
        except Exception as e:
            logger.error(f"  💥 ERROR: {e}")
            results.append([
                i, clean_text, "error", "error", "error", 0, 0, False, 0.0, 0.0, method_name, batch_size
            ])
    
    # Save results
    filename = save_results(results, method_name, batch_size)
    success_rate = (successful / len(texts)) * 100
    avg_requests = total_requests / len(texts)
    avg_perturbations = total_perturbations / len(texts)
    
    # Calculate averages for successful attacks only
    if semantic_similarities:
        avg_string_sim = np.mean(semantic_similarities)
        avg_pert_ratio = np.mean(perturbation_ratios)
    else:
        avg_string_sim = 0.0
        avg_pert_ratio = 0.0
    
    # Final summary
    logger.info("="*80)
    logger.info("🏁 COMPREHENSIVE BATCH ATTACK SUMMARY")
    logger.info("="*80)
    logger.info(f"📊 Method: {method_name} (batch_size={batch_size})")
    logger.info(f"✅ Success Rate: {successful}/{len(texts)} ({success_rate:.1f}%)")
    logger.info(f"🔍 Total Requests: {total_requests} (avg: {avg_requests:.1f} per example)")
    logger.info(f"🔄 Total Perturbations: {total_perturbations} (avg: {avg_perturbations:.1f} per example)")
    if successful > 0:
        logger.info(f"🎯 Avg String Similarity: {avg_string_sim:.3f}")
        logger.info(f"📈 Avg Perturbation Ratio: {avg_pert_ratio:.3f}")
    logger.info(f"💾 Results saved: {filename}")
    logger.info(f"📁 Log file: {log_filename}")
    logger.info("="*80)
    
    return {
        'success_rate': success_rate,
        'total_requests': total_requests,
        'avg_requests': avg_requests,
        'avg_string_similarity': avg_string_sim,
        'avg_perturbation_ratio': avg_pert_ratio,
        'results_file': str(filename),
        'log_file': log_filename
    }

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Batch TextBugger Test")
    parser.add_argument("--method", 
                       choices=["single_word", "multi_word", "adaptive_gradual", "adaptive_dynamic"], 
                       required=True,
                       help="Attack method (single_word for true 1-by-1, others for batch attacks)")
    parser.add_argument("--batch_size", type=int, default=3,
                       help="Batch size (default: 3)")
    parser.add_argument("--limit", type=int, default=100,
                       help="Number of examples to test (default: 100)")
    
    args = parser.parse_args()
    
    print("🎬 Starting Comprehensive Batch Attack Test")
    print(f"📊 Method: {args.method}")
    print(f"🔢 Batch Size: {args.batch_size}")
    print(f"📈 Examples: {args.limit}")
    print("🛑 Stops when: no more candidate words OR similarity threshold violated")
    print("="*60)
    
    try:
        results = run_batch_attack_comprehensive(args.method, args.batch_size, args.limit)
        
        print("✅ Test completed successfully!")
        print(f"📁 Check {results['results_file']} for detailed results")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()