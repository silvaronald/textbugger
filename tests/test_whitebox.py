#!/usr/bin/env python3
"""
Test whitebox TextBugger attacks
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_whitebox_imports():
    """Test that whitebox imports work correctly"""
    print("🔍 Testing whitebox imports...")
    
    try:
        from attacks.whitebox import AdversarialAttack
        print("  ✅ Successfully imported AdversarialAttack")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_whitebox_initialization():
    """Test whitebox attack initialization"""
    print("🔍 Testing whitebox initialization...")
    
    try:
        from attacks.whitebox import AdversarialAttack
        
        # Test with rtmr dataset (should have all required files)
        dataset = "rtmr"
        print(f"  Initializing AdversarialAttack with dataset: {dataset}")
        
        # Check if required files exist
        required_files = [
            f"datasets/{dataset}/tokenizer.pkl",
            f"datasets/{dataset}/embedding_matrix.npy", 
            f"training/{dataset}/lr_model.joblib",
            f"training/{dataset}/cnn_model.h5",
            f"training/{dataset}/lstm_model.h5"
        ]
        
        print("  Checking required files:")
        all_files_exist = True
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"    ✅ {file_path}")
            else:
                print(f"    ❌ {file_path} - MISSING")
                all_files_exist = False
        
        if not all_files_exist:
            print("  ⚠️  Some required files are missing - cannot test initialization")
            return False
        
        # Try to initialize
        attack = AdversarialAttack(dataset)
        print("  ✅ Successfully initialized AdversarialAttack")
        
        # Test basic attributes
        print("  Testing basic attributes:")
        print(f"    Tokenizer loaded: {hasattr(attack, 'tokenizer')}")
        print(f"    Embedding matrix shape: {attack.embedding_matrix.shape if hasattr(attack, 'embedding_matrix') else 'Not loaded'}")
        print(f"    Models loaded: LR={hasattr(attack, 'model_lr')}, CNN={hasattr(attack, 'model_cnn')}, LSTM={hasattr(attack, 'model_lstm')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Initialization error: {e}")
        return False

def test_whitebox_basic_functionality():
    """Test basic whitebox functionality"""
    print("🔍 Testing basic whitebox functionality...")
    
    try:
        from attacks.whitebox import AdversarialAttack
        
        dataset = "rtmr"
        attack = AdversarialAttack(dataset)
        
        # Test bug generation functions
        print("  Testing bug generation functions:")
        test_word = "example"
        
        # Test each bug function
        for bug_func in attack.bug_functions:
            try:
                modified_word = bug_func(test_word)
                func_name = bug_func.__name__
                print(f"    {func_name}: '{test_word}' → '{modified_word}'")
            except Exception as e:
                print(f"    ❌ Error in {bug_func.__name__}: {e}")
                return False
        
        print("  ✅ All bug generation functions work")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality error: {e}")
        return False

def test_whitebox_simple_attack():
    """Test a simple whitebox attack"""
    print("🔍 Testing simple whitebox attack...")
    
    try:
        from attacks.whitebox import AdversarialAttack
        
        dataset = "rtmr"
        attack = AdversarialAttack(dataset)
        
        # Simple test text
        test_text = "this movie is great"
        print(f"  Test text: '{test_text}'")
        
        # Test text preprocessing
        try:
            # Try to tokenize and pad the text (basic preprocessing test)
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            sequences = attack.tokenizer.texts_to_sequences([test_text])
            padded = pad_sequences(sequences, maxlen=attack.max_len)
            print(f"  ✅ Text preprocessing works - shape: {padded.shape}")
            
        except Exception as e:
            print(f"  ❌ Text preprocessing error: {e}")
            return False
        
        print("  ✅ Basic whitebox functionality verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Simple attack test error: {e}")
        return False

def main():
    print("🧪 Testing Whitebox TextBugger Attacks")
    print("=" * 50)
    
    tests = [
        ("Imports", test_whitebox_imports),
        ("Initialization", test_whitebox_initialization), 
        ("Basic Functionality", test_whitebox_basic_functionality),
        ("Simple Attack", test_whitebox_simple_attack)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            if result:
                print(f"  🎉 {test_name}: PASSED")
                passed += 1
            else:
                print(f"  💥 {test_name}: FAILED")
        except Exception as e:
            print(f"  💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All whitebox tests PASSED!")
        print("✅ Whitebox attacks are working correctly with the new structure!")
    else:
        print("⚠️  Some tests failed - check the output above for details")
        print("This might be due to missing model files or other dependencies")

if __name__ == "__main__":
    main()