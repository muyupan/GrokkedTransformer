import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

# def read_data_source_target(file_name, return_num=False, return_json=False):
#     """
#     file_name: a .json file containing a list of items, each has 'input_text', 'target_text', as keys
#     """
#     with open(file_name, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     if return_json:
#         if return_num:
#             return data, len(data)
#         return data

#     keys = list(data[0].keys())
#     source_target_pair = []
#     for item in data:
#         source_target_pair.append([item[key] for key in keys])

#     if return_num:
#         return pd.DataFrame(source_target_pair, columns=keys), len(data)
#     return pd.DataFrame(source_target_pair, columns=keys)

def compute_bridge_grounding_score(model_states, bridge_labels, layer_idx=5, k=1):
    """
    Compute bridge entity grounding score from intermediate layer states.
    
    Args:
        model_states: Hidden states from intermediate layers [batch, seq_len, hidden_dim]
        bridge_labels: Ground truth bridge entity IDs [batch]
        layer_idx: Which layer to extract bridge from (default 5, based on paper)
        k: Top-k for scoring (default 1)
    
    Returns:
        dict with 'bridge_grounding_score' and 'bridge_confidence'
    """
    # Extract states at r1 position (where bridge should be stored)
    bridge_states = model_states[layer_idx][:, -2, :]  # position r1
    
    # Project to vocabulary
    logits = model.lm_head(model.final_layer_norm(bridge_states))
    probs = F.softmax(logits, dim=-1)
    
    # Top-k accuracy
    top_k_preds = torch.topk(logits, k=k, dim=-1).indices
    correct = (top_k_preds == bridge_labels.unsqueeze(-1)).any(dim=-1).float()
    
    # Confidence (probability of correct bridge)
    confidence = probs.gather(1, bridge_labels.unsqueeze(-1)).squeeze()
    
    return {
        'bridge_grounding_score': correct.mean().item(),
        'bridge_confidence': confidence.mean().item(),
        'bridge_top1_accuracy': (top_k_preds[:, 0] == bridge_labels).float().mean().item()
    }


def read_data_source_target(file_path, return_num=False, return_json=False):
    """
    Read modular addition dataset and convert to OSU format
    
    Args:
        file_path: Path to train.json, valid.json, or test.json
        return_num: If True, return number of samples as second output
        return_json: If True, return original JSON (unused for modular arithmetic)
    
    Returns:
        DataFrame with 'input_text' and 'target_text' columns
    """
    import json
    import pandas as pd
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the data array
    if isinstance(data, dict) and 'data' in data:
        examples = data['data']
    else:
        examples = data
    
    # Convert to OSU format
    formatted_data = []
    for ex in examples:
        formatted_data.append({
            'input_text': ex['input'].strip(),
            'target_text': ex['output'].strip()
        })
    
    df = pd.DataFrame(formatted_data)
    
    if return_num:
        return df, len(df)
    else:
        return df

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute exact match accuracy for modular arithmetic
    
    Args:
        predictions: Model predictions (batch_size, vocab_size) logits or (batch_size,) indices
        labels: Ground truth labels (batch_size,)
    
    Returns:
        accuracy: Float between 0 and 1
    """
    if predictions.dim() > 1:
        # If logits, get argmax
        pred_indices = torch.argmax(predictions, dim=-1)
    else:
        pred_indices = predictions
    
    correct = (pred_indices == labels).float()
    accuracy = correct.mean().item()
    
    return accuracy

def compute_confidence_calibration(logits: torch.Tensor, 
                                   labels: torch.Tensor,
                                   n_bins: int = 10) -> Dict[str, float]:
    """
    Compute confidence calibration metrics
    
    Args:
        logits: Model output logits (batch_size, vocab_size)
        labels: Ground truth labels (batch_size,)
        n_bins: Number of bins for calibration curve
    
    Returns:
        Dictionary with calibration metrics
    """
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = (predictions == labels).float()
    
    # Expected Calibration Error (ECE)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.float().sum()
        
        if bin_size > 0:
            bin_acc = accuracies[in_bin].mean().item()
            bin_conf = confidences[in_bin].mean().item()
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            
            ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
    
    # Maximum Calibration Error (MCE)
    mce = max([abs(acc - conf) for acc, conf in zip(bin_accs, bin_confs) if conf > 0] or [0.0])
    
    # Confidence on correct vs incorrect predictions
    correct_mask = (predictions == labels)
    avg_correct_conf = confidences[correct_mask].mean().item() if correct_mask.sum() > 0 else 0.0
    
    incorrect_mask = ~correct_mask
    avg_incorrect_conf = confidences[incorrect_mask].mean().item() if incorrect_mask.sum() > 0 else 0.0
    
    return {
        'ece': ece.item() if isinstance(ece, torch.Tensor) else ece,
        'mce': mce,
        'avg_correct_confidence': avg_correct_conf,
        'avg_incorrect_confidence': avg_incorrect_conf,
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs
    }


def compute_consistency_score(logits_1: torch.Tensor, 
                              logits_2: torch.Tensor) -> float:
    """
    Compute consistency between two forward passes (for measuring model stability)
    
    Args:
        logits_1: First forward pass logits (batch_size, vocab_size)
        logits_2: Second forward pass logits (batch_size, vocab_size)
    
    Returns:
        consistency_score: Percentage of matching predictions
    """
    preds_1 = torch.argmax(logits_1, dim=-1)
    preds_2 = torch.argmax(logits_2, dim=-1)
    
    consistency = (preds_1 == preds_2).float().mean().item()
    
    return consistency


def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
    """
    Compute average entropy of attention distributions
    
    Args:
        attention_weights: Attention weights (n_layers, batch_size, n_heads, seq_len, seq_len)
                          or (batch_size, n_heads, seq_len, seq_len) for single layer
    
    Returns:
        Average entropy across all attention heads and positions
    """
    # Handle different input shapes
    if attention_weights.dim() == 4:
        # Single layer: (batch_size, n_heads, seq_len, seq_len)
        attn = attention_weights
    elif attention_weights.dim() == 5:
        # Multiple layers: (n_layers, batch_size, n_heads, seq_len, seq_len)
        # Average across layers
        attn = attention_weights.mean(dim=0)
    else:
        raise ValueError(f"Unexpected attention weights shape: {attention_weights.shape}")
    
    # Compute entropy: -sum(p * log(p))
    eps = 1e-10
    attn = attn + eps
    entropy = -(attn * torch.log(attn)).sum(dim=-1)  # Sum over attended positions
    
    # Average across all dimensions
    avg_entropy = entropy.mean().item()
    
    return avg_entropy


def compute_hallucination_metrics(logits: torch.Tensor,
                                  labels: torch.Tensor,
                                  confidence_threshold: float = 0.7) -> Dict[str, float]:
    """
    Compute hallucination-specific metrics
    
    Args:
        logits: Model output logits (batch_size, vocab_size)
        labels: Ground truth labels (batch_size,)
        confidence_threshold: Threshold for high confidence predictions
    
    Returns:
        Dictionary with hallucination metrics
    """
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = torch.max(probs, dim=-1)
    correct = (predictions == labels)
    
    # High confidence errors (hallucinations)
    high_conf_mask = confidences >= confidence_threshold
    high_conf_errors = high_conf_mask & (~correct)
    hallucination_rate = high_conf_errors.float().mean().item()
    
    # Accuracy on high confidence predictions
    high_conf_correct = high_conf_mask & correct
    high_conf_accuracy = high_conf_correct.float().sum() / (high_conf_mask.float().sum() + 1e-10)
    high_conf_accuracy = high_conf_accuracy.item()
    
    # Low confidence correct predictions
    low_conf_mask = confidences < confidence_threshold
    low_conf_correct = low_conf_mask & correct
    low_conf_accuracy = low_conf_correct.float().sum() / (low_conf_mask.float().sum() + 1e-10)
    low_conf_accuracy = low_conf_accuracy.item()
    
    return {
        'hallucination_rate': hallucination_rate,
        'high_conf_accuracy': high_conf_accuracy,
        'low_conf_accuracy': low_conf_accuracy,
        'high_conf_count': high_conf_mask.float().sum().item(),
        'low_conf_count': low_conf_mask.float().sum().item()
    }

def 