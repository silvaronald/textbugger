#!/usr/bin/env python3
"""
Unified TextBugger Attack Runner with FastWordBugger Support

This script consolidates all attack functionality:
- API-based blackbox attacks (IBM Watson, Azure, Google Cloud, AWS)
- Local model attacks (FastText, HuggingFace models)
- Support for both original TextBugger and optimized FastWordBugger
- Configurable datasets and attack limits

Usage:
    python scripts/run_attacks.py --target api --limit 10 --dataset rtmr
    python scripts/run_attacks.py --target local --model fasttext --dataset rtmr --limit 5
    
    # Train POS weights (recommended for better efficiency)
    python scripts/run_attacks.py --target api --limit 10 --dataset rtmr --train-pos
"""

import argparse
import os
import sys
import csv
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from attacks.fastwordbugger import FastWordBugger  # Import the optimized version
from models.api_models import APIModelWrapper
from clients.ibm_watson import IBMWatsonClassifier
from clients.azure_text import AzureTextAnalyticsClassifier
from clients.google_nlp import GoogleNLPClassifier
from clients.aws_comprehend import AWSComprehendClassifier
from utils.text_processing import split_sentences, split_words, fix_spacing

# Optional import for local models
try:
    from models.local_models import LocalModelWrapper
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging with timestamps"""
    import logging
    
    # Suppress verbose logging
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/textbugger_attacks_{timestamp}.log'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
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

def load_dataset(dataset_name, limit=None, from_initial=0):
    """Load dataset with optional limit and starting index"""
    base_path = Path(f"datasets/{dataset_name}")
    
    # Try different file names
    possible_files = [
        base_path / "X_test.csv",
        base_path / "test.csv"
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            df = pd.read_csv(file_path)
            texts = df["text"]
            
            # Apply from_initial offset
            if from_initial > 0:
                texts = texts.iloc[from_initial:]
            
            # Apply limit
            if limit:
                texts = texts.head(limit)
                
            return texts.tolist()
    
    raise FileNotFoundError(f"Could not find test data for {dataset_name}")

def load_training_data(dataset_name, limit=50):
    """Load training data for POS weight training"""
    base_path = Path(f"datasets/{dataset_name}")
    
    # Try different file names for training data
    possible_files = [
        base_path / "X_train.csv",
        base_path / "train.csv"
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            df = pd.read_csv(file_path)
            texts = df["text"]
            
            # Limit training data to avoid too many API calls
            if limit:
                texts = texts.head(limit)
                
            return texts.tolist()
    
    # If no training data found, use a subset of test data
    test_texts = load_dataset(dataset_name, limit=limit)
    return test_texts[:min(30, len(test_texts))]  # Use smaller subset from test

def save_results(results, target_type, model_name, dataset_name):
    """Save attack results to CSV"""
    results_dir = Path(f"results/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"{target_type}_{model_name}_FastWordBugger_attacks_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "original_text", "adversarial_text", "original_label", 
            "requests_used", "training_requests_used", "num_perturbed", "attack_successful", "timestamp"
        ])
        
        for result in results:
            writer.writerow(result + [datetime.now().isoformat()])
    
    return filename

def create_attacker(wrapper, training_data=None, logger=None):
    """Create FastWordBugger attacker"""
    attacker = FastWordBugger(wrapper, similarity_threshold=0.8, pos_threshold=0.1)
    
    # Train POS weights if training data provided
    if training_data:
        if logger:
            logger.info(f"🎓 Training POS weights with {len(training_data)} samples...")
        attacker.train_pos_weights(training_data)
        if logger:
            logger.info("✅ POS weights training completed")
    else:
        if logger:
            logger.info("⚠️ No training data provided - using default POS weights")
    
    return attacker

def run_single_attack(attacker, text, logger, model_name, attack_num, total_attacks):
    """Run a single attack and return results"""
    clean_text = re.sub(r'\s+', ' ', text).strip()
    logger.info(f"🎯 [{model_name}] Attack {attack_num}/{total_attacks}")
    
    try:
        adv_text, og_label, num_reqs, num_bugs, success = attacker.attack(clean_text)

        train_reqs = attacker.training_reqs if hasattr(attacker, 'training_reqs') else 0

        if success:
            logger.info(f"  ✅ SUCCESS - {num_reqs} requests, {num_bugs} bugs tested, {train_reqs} training requests used")
        else:
            logger.info(f"  ❌ FAILED - {num_reqs} requests, {num_bugs} bugs tested, {train_reqs} training requests used")

        return [clean_text, adv_text, og_label, num_reqs, train_reqs, num_bugs, success], num_reqs, success
        
    except Exception as e:
        logger.error(f"  💥 ERROR: {e}")
        return [clean_text, "error", "error", 0, 0, False], 0, False

def run_api_attacks(models, dataset_name, limit, train_pos, logger, from_initial=0):
    """Run FastWordBugger attacks on API models"""
    api_clients = {}
    
    # Try to initialize each client
    try:
        api_clients["ibm_watson"] = IBMWatsonClassifier()
    except Exception as e:
        logger.warning(f"⚠️ IBM Watson not available: {e}")
    
    # try:
    #     api_clients["azure_text_analytics"] = AzureTextAnalyticsClassifier()
    # except Exception as e:
    #     logger.warning(f"⚠️ Azure Text Analytics not available: {e}")
    
    try:
        api_clients["google_cloud_nlp"] = GoogleNLPClassifier()
    except Exception as e:
        logger.warning(f"⚠️ Google Cloud NLP not available: {e}")
    
    try:
        api_clients["aws_comprehend"] = AWSComprehendClassifier()
    except Exception as e:
        logger.warning(f"⚠️ AWS Comprehend not available: {e}")
    
    if models:
        # Filter to requested models
        api_clients = {k: v for k, v in api_clients.items() if k in models}
    
    texts = load_dataset(dataset_name, limit, from_initial)
    
    # Load training data for POS weights if needed
    training_data = None
    if train_pos:
        try:
            training_data = load_training_data(dataset_name, limit=30)
            logger.info(f"📚 Loaded {len(training_data)} training samples for POS weight calculation")
        except Exception as e:
            logger.warning(f"⚠️ Could not load training data: {e}")
    
    overall_results = {}
    
    for api_name, client in api_clients.items():
        logger.info(f"🚀 Starting {api_name} FastWordBugger attacks on {dataset_name}")
        
        try:
            wrapper = APIModelWrapper(client)
            attacker = create_attacker(wrapper, training_data, logger)
            
            results = []
            successful = 0
            total_requests = 0
            
            for i, text in enumerate(texts, 1):
                result, num_reqs, success = run_single_attack(
                    attacker, text, logger, api_name, i, len(texts)
                )
                
                if success:
                    successful += 1
                
                total_requests += num_reqs
                results.append(result)
            
            # Save results
            filename = save_results(results, "api", api_name, dataset_name)
            success_rate = (successful / len(texts)) * 100
            avg_requests = total_requests / len(texts) if len(texts) > 0 else 0
            
            logger.info(f"📊 {api_name} Summary:")
            logger.info(f"  Success: {successful}/{len(texts)} ({success_rate:.1f}%)")
            logger.info(f"  Requests: {total_requests} total, {avg_requests:.1f} avg per attack")
            logger.info(f"💾 Results: {filename}")
            
            overall_results[api_name] = {
                'successful': successful,
                'total': len(texts),
                'success_rate': success_rate,
                'total_requests': total_requests,
                'avg_requests': avg_requests,
                'results_file': str(filename)
            }
                
        except Exception as e:
            logger.error(f"💥 Failed {api_name}: {e}")
    
    return overall_results

def run_local_attacks(models, dataset_name, limit, train_pos, logger, from_initial=0):
    """Run FastWordBugger attacks on local models"""
    if not LOCAL_MODELS_AVAILABLE:
        logger.error("❌ Local models not available - missing dependencies (fasttext, transformers, etc.)")
        logger.info("💡 Install with: pip install fasttext transformers torch")
        return {}
    
    if not models:
        models = ["fasttext", "sst2", "toxic-bert", "twitter-roberta"]
    
    texts = load_dataset(dataset_name, limit, from_initial)
    
    # Load training data for POS weights if needed
    training_data = None
    if train_pos:
        try:
            training_data = load_training_data(dataset_name, limit=30)
            logger.info(f"📚 Loaded {len(training_data)} training samples for POS weight calculation")
        except Exception as e:
            logger.warning(f"⚠️ Could not load training data: {e}")
    
    overall_results = {}
    
    for model_name in models:
        logger.info(f"🚀 Starting {model_name} FastWordBugger attacks on {dataset_name}")
        
        try:
            wrapper = LocalModelWrapper(model_name)
            attacker = create_attacker(wrapper.classify, training_data, logger)
            
            results = []
            successful = 0
            total_requests = 0
            
            for i, text in enumerate(texts, 1):
                result, num_reqs, success = run_single_attack(
                    attacker, text, logger, model_name, i, len(texts)
                )
                
                if success:
                    successful += 1
                
                total_requests += num_reqs
                results.append(result)
            
            # Save results
            filename = save_results(results, "local", model_name, dataset_name)
            success_rate = (successful / len(texts)) * 100
            avg_requests = total_requests / len(texts) if len(texts) > 0 else 0
            
            logger.info(f"📊 {model_name} Summary:")
            logger.info(f"  Success: {successful}/{len(texts)} ({success_rate:.1f}%)")
            logger.info(f"  Requests: {total_requests} total, {avg_requests:.1f} avg per attack")
            logger.info(f"💾 Results: {filename}")
            
            overall_results[model_name] = {
                'successful': successful,
                'total': len(texts),
                'success_rate': success_rate,
                'total_requests': total_requests,
                'avg_requests': avg_requests,
                'results_file': str(filename)
            }
                
        except Exception as e:
            logger.error(f"💥 Failed {model_name}: {e}")
    
    return overall_results

def main():
    parser = argparse.ArgumentParser(description="Run FastWordBugger attacks")
    parser.add_argument("--target", choices=["api", "local"], required=True,
                       help="Target type: api (API models) or local (local models)")
    parser.add_argument("--model", nargs="*", 
                       help="Specific models to test (optional)")
    parser.add_argument("--dataset", default="rtmr", 
                       help="Dataset to use (default: rtmr)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples (default: use NUM_ATTACKS_API from .env)")
    parser.add_argument("--from-initial", type=int, default=0,
                       help="Start from this index in the dataset (default: 0)")
    parser.add_argument("--train-pos", action="store_true",
                       help="Train POS weights using training data (recommended for better efficiency)")
    
    args = parser.parse_args()
    
    # Setup
    logger, log_filename = setup_logging()
    
    # Get limit from env if not specified
    if args.limit is None:
        args.limit = int(os.getenv('NUM_ATTACKS_API', 5))
    
    logger.info("🎬 Starting FastWordBugger Attacks")
    logger.info(f"Target: {args.target}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Limit: {args.limit}")
    logger.info(f"Models: {args.model or 'all'}")
    logger.info(f"From initial index: {args.from_initial}")
    logger.info(f"Train POS weights: {args.train_pos}")
    logger.info("="*80)
    
    # Run attacks
    if args.target == "api":
        results = run_api_attacks(
            args.model, args.dataset, args.limit,
            args.train_pos, logger, from_initial=args.from_initial
        )
    else:
        results = run_local_attacks(
            args.model, args.dataset, args.limit,
            args.train_pos, logger, from_initial=args.from_initial
        )
    
    # Final summary
    logger.info("="*80)
    logger.info("🏁 FINAL SUMMARY")
    logger.info("="*80)
    
    for model_name, stats in results.items():
        logger.info(f"🔹 {model_name}: {stats['successful']}/{stats['total']} "
                   f"({stats['success_rate']:.1f}%) - {stats['total_requests']} requests "
                   f"(avg: {stats['avg_requests']:.1f})")
    
    logger.info("✅ All attacks completed!")
    logger.info(f"📁 Check results/ folder and {log_filename} for details")

if __name__ == "__main__":
    main()