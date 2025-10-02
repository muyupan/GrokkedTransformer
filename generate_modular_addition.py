"""
generate_modular_addition.py
Generates modular addition dataset for grokking experiments
Compatible with OSU GrokkedTransformer code
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Tuple
import pickle

class ModularAdditionDatasetGenerator:
    def __init__(self, 
                 p: int = 97,
                 train_fraction: float = 0.3,
                 seed: int = 42,
                 output_dir: str = './data/modular_addition'):
        """
        Generate modular addition dataset for grokking
        
        Args:
            p: Prime modulus (97 is standard)
            train_fraction: Fraction for training (0.3 for grokking)
            seed: Random seed
            output_dir: Where to save the dataset
        """
        self.p = p
        self.train_fraction = train_fraction
        self.seed = seed
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_all_examples(self) -> List[Dict]:
        """Generate all possible (a + b) mod p examples"""
        examples = []
        
        for a in range(self.p):
            for b in range(self.p):
                result = (a + b) % self.p
                
                # Multiple format options
                example = {
                    # Format 1: Question-answer style
                    'question': f'{a}+{b}=',
                    'answer': str(result),
                    
                    # Format 2: Full equation
                    'equation': f'{a}+{b}={result}',
                    
                    # Format 3: With spaces (GPT-2 friendly)
                    'input': f'{a} + {b} =',
                    'output': f' {result}',
                    
                    # Format 4: Text format
                    'text': f'{a} + {b} = {result}',
                    
                    # Metadata
                    'a': a,
                    'b': b,
                    'result': result,
                    'operation': 'addition',
                    'modulus': self.p
                }
                examples.append(example)
        
        return examples
    
    def split_dataset(self, examples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split into train and test sets"""
        # Shuffle examples
        random.shuffle(examples)
        
        # Calculate split point
        n_train = int(len(examples) * self.train_fraction)
        
        train_data = examples[:n_train]
        test_data = examples[n_train:]
        
        return train_data, test_data
    
    def save_json_format(self, train_data: List[Dict], test_data: List[Dict]):
        """Save in JSON format (human-readable)"""
        # Save train data
        train_path = os.path.join(self.output_dir, 'train.json')
        with open(train_path, 'w') as f:
            json.dump({
                'data': train_data,
                'metadata': {
                    'size': len(train_data),
                    'modulus': self.p,
                    'operation': 'addition',
                    'train_fraction': self.train_fraction
                }
            }, f, indent=2)
        
        # Save test data
        test_path = os.path.join(self.output_dir, 'test.json')
        with open(test_path, 'w') as f:
            json.dump({
                'data': test_data,
                'metadata': {
                    'size': len(test_data),
                    'modulus': self.p,
                    'operation': 'addition',
                    'test_fraction': 1 - self.train_fraction
                }
            }, f, indent=2)
        
        print(f"Saved JSON format to {self.output_dir}")
    
    def save_text_format(self, train_data: List[Dict], test_data: List[Dict]):
        """Save in simple text format (one equation per line)"""
        # Train file
        train_path = os.path.join(self.output_dir, 'train.txt')
        with open(train_path, 'w') as f:
            for ex in train_data:
                f.write(f"{ex['text']}\n")
        
        # Test file
        test_path = os.path.join(self.output_dir, 'test.txt')
        with open(test_path, 'w') as f:
            for ex in test_data:
                f.write(f"{ex['text']}\n")
        
        print(f"Saved text format to {self.output_dir}")
    
    def save_pytorch_format(self, train_data: List[Dict], test_data: List[Dict]):
        """Save in PyTorch-ready format"""
        import torch
        
        def create_tensors(data):
            inputs = [ex['input'] for ex in data]
            outputs = [ex['output'] for ex in data]
            return {
                'inputs': inputs,
                'outputs': outputs,
                'full_texts': [ex['text'] for ex in data],
                'metadata': {
                    'a_values': torch.tensor([ex['a'] for ex in data]),
                    'b_values': torch.tensor([ex['b'] for ex in data]),
                    'results': torch.tensor([ex['result'] for ex in data])
                }
            }
        
        # Save train
        train_dict = create_tensors(train_data)
        torch.save(train_dict, os.path.join(self.output_dir, 'train.pt'))
        
        # Save test
        test_dict = create_tensors(test_data)
        torch.save(test_dict, os.path.join(self.output_dir, 'test.pt'))
        
        print(f"Saved PyTorch format to {self.output_dir}")
    
    def create_osu_compatible_dataset(self, train_data: List[Dict], test_data: List[Dict]):
        """
        Create dataset compatible with OSU GrokkedTransformer code
        Based on their expected format
        """
        # Create a format similar to their datasets
        dataset = {
            'train': {
                'data': [],
                'labels': []
            },
            'test': {
                'data': [],
                'labels': []
            },
            'val': {  # Use part of test as validation
                'data': [],
                'labels': []
            }
        }
        
        # Process train data
        for ex in train_data:
            dataset['train']['data'].append(ex['input'])
            dataset['train']['labels'].append(ex['output'])
        
        # Split test into test and val (50-50)
        val_size = len(test_data) // 2
        val_data = test_data[:val_size]
        final_test_data = test_data[val_size:]
        
        # Process validation data
        for ex in val_data:
            dataset['val']['data'].append(ex['input'])
            dataset['val']['labels'].append(ex['output'])
        
        # Process test data
        for ex in final_test_data:
            dataset['test']['data'].append(ex['input'])
            dataset['test']['labels'].append(ex['output'])
        
        # Save as pickle (OSU format)
        pickle_path = os.path.join(self.output_dir, 'modular_addition_p97.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Saved OSU-compatible format to {pickle_path}")
        
        # Also save metadata
        metadata = {
            'dataset_name': f'modular_addition_p{self.p}',
            'train_size': len(dataset['train']['data']),
            'val_size': len(dataset['val']['data']),
            'test_size': len(dataset['test']['data']),
            'total_possible': self.p * self.p,
            'train_fraction': self.train_fraction,
            'modulus': self.p,
            'vocab_size': self.p,  # Output vocab is 0 to p-1
            'seed': self.seed
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return dataset
    
    def generate_and_save_all(self):
        """Generate and save dataset in all formats"""
        print(f"Generating modular addition dataset (mod {self.p})")
        print(f"Train fraction: {self.train_fraction}")
        
        # Generate all examples
        all_examples = self.generate_all_examples()
        print(f"Total examples: {len(all_examples)}")
        
        # Split into train/test
        train_data, test_data = self.split_dataset(all_examples)
        print(f"Train examples: {len(train_data)}")
        print(f"Test examples: {len(test_data)}")
        
        # Save in multiple formats
        self.save_json_format(train_data, test_data)
        self.save_text_format(train_data, test_data)
        self.save_pytorch_format(train_data, test_data)
        dataset = self.create_osu_compatible_dataset(train_data, test_data)
        
        # Print sample examples
        print("\nSample examples:")
        for i in range(min(5, len(train_data))):
            ex = train_data[i]
            print(f"  {ex['input']} {ex['output']}")
        
        return dataset


# PyTorch Dataset Class (for use with DataLoader)
import torch
from torch.utils.data import Dataset

class ModularAdditionDataset(Dataset):
    """PyTorch Dataset for modular addition"""
    
    def __init__(self, data_path: str, split: str = 'train'):
        """
        Args:
            data_path: Path to the data directory
            split: 'train', 'test', or 'val'
        """
        # Load the pickle file
        pkl_path = os.path.join(data_path, 'modular_addition_p97.pkl')
        with open(pkl_path, 'rb') as f:
            self.data_dict = pickle.load(f)
        
        self.split = split
        self.inputs = self.data_dict[split]['data']
        self.outputs = self.data_dict[split]['labels']
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'output': self.outputs[idx],
            'text': self.inputs[idx] + self.outputs[idx]
        }


# Integration with OSU code
def integrate_with_osu_code():
    """
    Example of how to integrate with OSU GrokkedTransformer
    """
    code = '''
    # In your training script, add:
    
    from generate_modular_addition import ModularAdditionDataset
    
    # Load dataset
    train_dataset = ModularAdditionDataset('./data/modular_addition', split='train')
    val_dataset = ModularAdditionDataset('./data/modular_addition', split='val')
    test_dataset = ModularAdditionDataset('./data/modular_addition', split='test')
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    '''
    
    return code


if __name__ == "__main__":
    # Generate the dataset
    generator = ModularAdditionDatasetGenerator(
        p=97,                    # Standard modulus
        train_fraction=0.3,      # 30% train for grokking
        seed=42,                 # Reproducible
        output_dir='./data/modular_addition'
    )
    
    # Generate and save everything
    dataset = generator.generate_and_save_all()
    
    print("\n" + "="*50)
    print("Dataset generation complete!")
    print("="*50)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Modulus (p): {generator.p}")
    print(f"Total possible examples: {generator.p * generator.p}")
    print(f"Train examples: {len(dataset['train']['data'])}")
    print(f"Validation examples: {len(dataset['val']['data'])}")
    print(f"Test examples: {len(dataset['test']['data'])}")
    print(f"Train fraction: {generator.train_fraction:.1%}")
    
    # Show how to use with OSU code
    print("\nTo use with OSU code:")
    print(integrate_with_osu_code())