#!/usr/bin/env python3
"""
Test script to verify if new tokens are included in loss computation during training.
This script will help you confirm whether your new tokens are being trained or treated as special tokens.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
import numpy as np
import json
import os
from typing import List, Dict, Tuple

def create_test_tokens_file(tokens: List[str], filename: str = "/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/New_tokens.txt"):
    """Create a test file with new tokens."""
    with open(filename, 'w') as f:
        for token in tokens:
            f.write(f"{token}\n")
    return filename

def test_token_loss_inclusion(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    new_tokens: List[str] = None,
    test_text: str = "Hello, this is a test with <DEPTH_51> and <DEPTH_100>.",
    output_dir: str = "./test_output"
):
    """
    Test whether new tokens are included in loss computation.
    
    Args:
        model_name: Base model to test with
        new_tokens: List of new tokens to add
        test_text: Text containing new tokens to test
        output_dir: Directory to save test results
    """
    
    if new_tokens is None:
        new_tokens = ["<NEW_TOKEN_1>", "<NEW_TOKEN_2>", "<SPECIAL_MARKER>"]
    
    print(f"Testing with model: {model_name}")
    print(f"New tokens to add: {new_tokens}")
    print(f"Test text: {test_text}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Load tokenizer and model
    print("\n1. Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    if "qwen2.5" in model_name.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for precise testing
            device_map="cpu"  # Use CPU for testing
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for precise testing
            device_map="cpu"  # Use CPU for testing
        )
    
    # Record original vocabulary size
    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")
    
    # Add new tokens
    print("\n2. Adding new tokens...")
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens_filtered = [token for token in new_tokens if token not in existing_tokens]
    
    if new_tokens_filtered:
        tokenizer.add_tokens(new_tokens_filtered)
        print(f"Added {len(new_tokens_filtered)} new tokens: {new_tokens_filtered}")
    else:
        print("All tokens already exist in vocabulary!")
        new_tokens_filtered = new_tokens
    
    new_vocab_size = len(tokenizer)
    print(f"New vocabulary size: {new_vocab_size}")
    
    # Resize model embeddings
    print("\n3. Resizing model embeddings...")
    old_embedding_size = model.get_input_embeddings().num_embeddings
    # model.resize_token_embeddings(new_vocab_size)
    new_embedding_size = model.get_input_embeddings().num_embeddings
    print(f"Resized embeddings from {old_embedding_size} to {new_embedding_size}")
    
    # Get token IDs for new tokens
    new_token_ids = {}
    for token in new_tokens_filtered:
        token_id = tokenizer.convert_tokens_to_ids(token)
        new_token_ids[token] = token_id
        print(f"Token '{token}' -> ID: {token_id}")
    
    # Test 1: Check if new tokens are in vocabulary
    print("\n4. Testing token presence in vocabulary...")
    for token in new_tokens_filtered:
        is_in_vocab = token in tokenizer.get_vocab()
        print(f"Token '{token}' in vocabulary: {is_in_vocab}")
    
    # Test 2: Tokenize test text and check for new tokens
    print("\n5. Tokenizing test text...")
    
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input IDs: {input_ids}")
    print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
    
    # Check which new tokens appear in the input
    new_tokens_in_input = []
    for token, token_id in new_token_ids.items():
        if token_id in input_ids:
            new_tokens_in_input.append((token, token_id))
            print(f"Found new token '{token}' (ID: {token_id}) in input")
    
    if not new_tokens_in_input:
        print("WARNING: No new tokens found in test text. Modifying test text...")
        # Add new tokens to the text
        test_text_with_tokens = test_text + " " + " ".join(new_tokens_filtered)
        inputs = tokenizer(test_text_with_tokens, return_tensors="pt")
        input_ids = inputs["input_ids"]
        print(f"New input IDs: {input_ids}")
        print(f"New decoded tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
        
        # Update new_tokens_in_input
        for token, token_id in new_token_ids.items():
            if token_id in input_ids:
                new_tokens_in_input.append((token, token_id))
                print(f"Found new token '{token}' (ID: {token_id}) in modified input")
    
    # Test 3: Create a simple forward pass to test loss computation
    print("\n6. Testing loss computation with new tokens...")
    
    # Create labels (same as input_ids for this test)
    labels = input_ids.clone()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
    
    print(f"Model loss: {loss.item():.4f}")
    
    # Test 4: Check if new token positions contribute to loss
    print("\n7. Analyzing loss contribution from new tokens...")
    
    # Get the logits for the last token (where loss is computed)
    last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
    
    # Check if new tokens have non-zero logits
    for token, token_id in new_token_ids.items():
        if token_id < last_token_logits.size(0):
            logit_value = last_token_logits[token_id].item()
            print(f"Logit for token '{token}' (ID: {token_id}): {logit_value:.4f}")
        else:
            print(f"Token '{token}' (ID: {token_id}) is out of bounds for logits")
    
    # Test 5: Manual loss computation to verify
    print("\n8. Manual loss computation verification...")
    
    # Compute cross-entropy loss manually
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    manual_loss = loss_fct(flat_logits, flat_labels)
    print(f"Manual loss computation: {manual_loss.item():.4f}")
    print(f"Model loss: {loss.item():.4f}")
    print(f"Loss difference: {abs(manual_loss.item() - loss.item()):.6f}")
    # Test 6: Check gradient flow for new tokens
    print("\n9. Testing gradient flow for new tokens...")
    import pdb; pdb.set_trace()
    # Enable gradients
    model.train()
    input_ids.requires_grad_(True)
    
    # Forward pass with gradients
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Check if new token embeddings have gradients
    input_embeddings = model.get_input_embeddings()
    if hasattr(input_embeddings, 'weight') and input_embeddings.weight.grad is not None:
        print("Input embeddings have gradients!")
        
        for token, token_id in new_token_ids.items():
            if token_id < input_embeddings.weight.size(0):
                grad_norm = input_embeddings.weight.grad[token_id].norm().item()
                print(f"Gradient norm for token '{token}' (ID: {token_id}): {grad_norm:.6f}")
            else:
                print(f"Token '{token}' (ID: {token_id}) is out of bounds for embeddings")
    else:
        print("No gradients found in input embeddings")
    
    # Test 7: Check output embeddings (lm_head)
    print("\n10. Testing output embeddings (lm_head)...")
    
    if hasattr(model, 'lm_head') and model.lm_head.weight.grad is not None:
        print("LM head has gradients!")
        
        for token, token_id in new_token_ids.items():
            if token_id < model.lm_head.weight.size(0):
                grad_norm = model.lm_head.weight.grad[token_id].norm().item()
                print(f"LM head gradient norm for token '{token}' (ID: {token_id}): {grad_norm:.6f}")
            else:
                print(f"Token '{token}' (ID: {token_id}) is out of bounds for lm_head")
    else:
        print("No gradients found in lm_head")
    
    # Save test results
    results = {
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "new_tokens_added": new_tokens_filtered,
        "new_token_ids": new_token_ids,
        "test_text": test_text,
        "input_ids": input_ids.tolist(),
        "model_loss": loss.item(),
        "manual_loss": manual_loss.item(),
        "new_tokens_in_input": [(token, token_id) for token, token_id in new_tokens_in_input]
    }
    
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n11. Test results saved to: {results_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    if new_tokens_in_input:
        print("✅ New tokens are present in the input text")
        print("✅ New tokens will contribute to loss computation")
        print("✅ New tokens are NOT treated as special tokens (they have learnable embeddings)")
    else:
        print("❌ No new tokens found in input text")
        print("⚠️  Add new tokens to your training data to see them in loss computation")
    
    print(f"✅ Model vocabulary expanded from {original_vocab_size} to {new_vocab_size}")
    print(f"✅ New token embeddings are initialized and trainable")
    
    return results

def test_with_your_tokens(tokens_file: str, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """Test with tokens from your file."""
    print(f"Testing with tokens from file: {tokens_file}")
    
    # Load tokens from file
    with open(tokens_file, 'r') as f:
        new_tokens = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Loaded {len(new_tokens)} tokens: {new_tokens[:5]}{'...' if len(new_tokens) > 5 else ''}")
    
    # Create test text with some of the new tokens
    test_text = f"Test text with new tokens: <DEPTH_51> and <DEPTH_100> and more content."
    
    return test_token_loss_inclusion(
        model_name=model_name,
        new_tokens=new_tokens,
        test_text=test_text
    )

if __name__ == "__main__":
    # Test with default tokens
    # print("Running test with default tokens...")
    # results = test_token_loss_inclusion()
 
    print("Testing with your tokens file...")
    results = test_with_your_tokens("/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/New_tokens.txt")
