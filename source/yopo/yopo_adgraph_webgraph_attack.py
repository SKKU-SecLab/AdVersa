#!/usr/bin/env python3
"""
YOPO Attack Script for AdGraph and WebGraph Models
Evaluates attack performance on simpler baseline models.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import os
import yaml
import warnings
import pickle

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*incompatible dtype.*')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from cost_dicts import get_cost_dict

# Model paths
ADGRAPH_MODEL_PATH = '../../models/sota_comparison/AdGraph.joblib'
WEBGRAPH_MODEL_PATH = '../../models/sota_comparison/WebGraph.joblib'
FEATURES_PATH = '../../models/features.yaml'

class GraphDataset(Dataset):
    """Custom Dataset for AdGraph/WebGraph data"""
    def __init__(self, data_dir, feature_columns):
        # Load parquet or CSV file
        self.data=[]
        for datafile in os.listdir(data_dir):
            if datafile.endswith('.parquet'):
                self.data.append(pd.read_parquet(data_dir+"/"+datafile))
            else:
                self.data.append(pd.read_csv(data_dir+"/"+datafile))
        self.data=pd.concat(self.data)

        # Filter to only include specified features
        available_features = [f for f in feature_columns if f in self.data.columns]
        label_col = 'label' if 'label' in self.data.columns else 'CLASS'

        if label_col in self.data.columns:
            self.data = self.data[available_features + [label_col]]
        else:
            self.data = self.data[available_features]

        # Handle label column
        if label_col in self.data.columns:
            self.features = self.data.drop(label_col, axis=1).values.astype(np.float32)
            self.labels = self.data[label_col].values.astype(np.int64)
        else:
            self.features = self.data.values.astype(np.float32)
            self.labels = np.zeros(len(self.features), dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

class SurrogateModel(nn.Module):
    """Surrogate neural network model (matching original YOPO exactly)"""
    def __init__(self, input_size):
        super(SurrogateModel, self).__init__()
        # Original YOPO architecture: 1024 -> 512 -> 256 -> 2
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Original YOPO: NO dropout during forward pass (lines 71-76 in surrogate_cross.py)
        x = self.leaky_relu(self.linear1(x))
        # NO dropout applied
        x = self.leaky_relu(self.linear2(x))
        # NO dropout applied
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        return x

def load_feature_list(yaml_path, model_type):
    """Load feature list from YAML config"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    if model_type == 'adgraph':
        features = config['model_AdGraph']
    else:  # webgraph
        features = config['model_WebGraph']

    # Remove metadata columns
    # features = [f for f in features if f not in ['visit_id', 'name', 'domain', 'top_level_domain']]
    return features

def load_sklearn_model_compat(model_path):
    """Load sklearn model with version compatibility"""
    import joblib
    from sklearn import __version__ as sklearn_version

    print(f"Current sklearn version: {sklearn_version}")
    print(f"NOTE: Models trained with sklearn 1.1.1, loading with {sklearn_version}")
    print(f"      This may cause compatibility issues. Predictions should still work.")

    try:
        # Try to load - sklearn should handle prediction even if internal format differs
        model = joblib.load(model_path)
        print(f"    Model loaded: {type(model)}")
        return model

    except Exception as e:
        error_msg = str(e)
        # Check for various sklearn incompatibility errors
        if any(x in error_msg for x in ["incompatible dtype", "node array", "invalid load key"]):
            print(f"\n    *** SKLEARN VERSION MISMATCH ***")
            print(f"    The models were trained with sklearn 1.1.1 but you have {sklearn_version}")
            print(f"    Error: {error_msg[:150]}...")
            print(f"\n    WORKAROUND OPTIONS:")
            print(f"    1. Install sklearn 1.1.1 in a separate environment with Python 3.9-3.10")
            print(f"    2. Retrain the models with current sklearn version")
            print(f"    3. Export model to ONNX/PMML format for compatibility")
            print(f"\n    For now, skipping sklearn model evaluation.")
            print(f"    Surrogate model attacks will still run.\n")
            return None  # Return None to skip sklearn evaluation
        else:
            # For other errors, raise them
            raise RuntimeError(f"Could not load model from {model_path}: {e}")

def get_feature_types(feature_list):
    """Identify binary and numerical features"""
    binary_features = []
    numerical_features = []
    increase_only_features = []

    for i, feat_name in enumerate(feature_list):
        # Binary features: typically have 'is_', '_present', or are known binary
        if any(x in feat_name for x in ['is_', '_present', 'keyword_', 'ad_size', 'screen_size',
                                         'base_domain', 'semicolon', 'ascendant_has']):
            binary_features.append(i)
        else:
            numerical_features.append(i)

        # Increase-only features (counts, sizes - should not decrease)
        if any(x in feat_name for x in ['num_', 'length', 'size', 'count', 'depth', 'breadth']):
            increase_only_features.append(i)

    return binary_features, numerical_features, increase_only_features

def inject_feature(features, perturbation, feature_list):
    """Inject perturbation into features following original YOPO logic"""
    result = features.clone()
    binary_features, numerical_features, _ = get_feature_types(feature_list)

    # For numerical features: direct addition
    for num_idx in numerical_features:
        result[:, num_idx] += perturbation[:, num_idx]

    # For binary features: add perturbation and apply sigmoid to keep in [0,1]
    for bin_idx in binary_features:
        result[:, bin_idx] += perturbation[:, bin_idx]
        # Apply sigmoid to constrain to [0, 1] range (similar to original softmax logic)
        result[:, bin_idx] = torch.sigmoid(result[:, bin_idx])

    return result

def pert_constraints(pert, feature_list, final_step=False):
    """Apply perturbation constraints following original YOPO logic"""
    binary_features, numerical_features, increase_only_features = get_feature_types(feature_list)

    # For increase-only features: ensure perturbation is non-negative
    for increase_idx in increase_only_features:
        if pert[increase_idx] < 0:
            pert[increase_idx] = 0

    # For binary features in final step: round to 0 or 1
    if final_step:
        for bin_idx in binary_features:
            # Round to nearest integer (0 or 1)
            pert[bin_idx] = torch.round(pert[bin_idx])

    return pert

def compute_cost_graph(injected_features, orig_features, feature_list, model_type, cost_type='DC'):
    """Compute cost using original YOPO cost dictionaries"""
    cost = torch.tensor(0.0, device=injected_features.device)
    diff_features = injected_features - orig_features

    # Get cost dictionary from original YOPO
    cost_dict = get_cost_dict(model_type, cost_type)

    # Default cost for features not in dictionary
    DEFAULT_COST = 1.0

    # Calculate weighted cost
    for i, feature_name in enumerate(feature_list):
        if feature_name in cost_dict:
            weight = cost_dict[feature_name]
        else:
            # Default cost for unknown features
            weight = DEFAULT_COST

        cost += weight * diff_features[:, i].abs().sum()

    return cost / diff_features.shape[0]  # Divide by batch size like original YOPO

def uap_attack(model, features, labels, feature_list, model_type, epsilon=10, step_size=0.1,
               num_steps=100, lagrangian=400, cost_type='DC', device='cpu'):
    """
    Universal Adversarial Perturbation attack following original YOPO implementation

    Args:
        model: Trained surrogate model
        features: Input features tensor
        labels: Ground truth labels tensor
        feature_list: List of feature names
        model_type: Model type ('adgraph' or 'webgraph')
        epsilon: Maximum perturbation magnitude
        step_size: Step size for optimization
        num_steps: Number of optimization steps
        lagrangian: Lagrangian multiplier for cost constraint
        cost_type: Cost model type ('DC', 'HSC', 'HCC')
        device: Device to run on
    """

    # Initialize perturbation (following original YOPO)
    p = torch.zeros_like(features[0]).to(device)
    p.requires_grad_(True)

    features = features.to(device)
    labels = labels.to(device)
    train_size = features.shape[0]

    print(f"Starting UAP attack")

    # Get feature bounds for clamping - following original YOPO logic
    max_values = features.max(dim=0)[0]
    min_values = features.min(dim=0)[0]
    diff_values = max_values - min_values

    # YOPO logic: numerical features use natural range, non-numerical use epsilon
    # For now, assume most features are numerical (like original), use epsilon for binary/categorical
    epsilon_tensor = torch.clone(diff_values)

    # For features that appear to be binary/categorical (small range), use epsilon
    binary_cat_count = 0
    for i in range(len(diff_values)):
        if diff_values[i] <= 1.0:  # Likely binary/categorical feature
            epsilon_tensor[i] = epsilon
            binary_cat_count += 1

    print(f"Feature bounds analysis:")
    print(f"  Total features: {len(diff_values)}")
    print(f"  Binary/categorical features (range ≤ 1.0): {binary_cat_count}")
    print(f"  Numerical features: {len(diff_values) - binary_cat_count}")
    print(f"  Epsilon tensor min: {epsilon_tensor.min().item():.3f}")
    print(f"  Epsilon tensor max: {epsilon_tensor.max().item():.3f}")

    # Use Adam optimizer like original YOPO
    print(f"  Using Adam optimizer with lr={step_size}")

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam([p], lr=step_size)

    for step in range(num_steps):
        final_step = (step == num_steps - 1)

        # Expand perturbation to match batch size (following original YOPO)
        p_exp = p.repeat(train_size, 1)

        # Inject perturbation into features
        injected_features = inject_feature(features, p_exp, feature_list)

        # Forward pass
        outputs = model(injected_features)

        # Compute cost (following original YOPO cost computation)
        cost = compute_cost_graph(injected_features, features, feature_list, model_type, cost_type)

        # Calculate loss (following original YOPO: cost - lagrangian * classification_loss)
        classification_loss = loss_fn(outputs, labels)
        loss = cost - (lagrangian * classification_loss)

        # Backpropagation and optimization (following original YOPO)
        optimizer.zero_grad()
        loss.backward()

        # Apply gradient sign and step (original YOPO uses sign of gradient)
        with torch.no_grad():
            if p.grad is not None:
                p.grad.sign_()

        # Optimizer step
        optimizer.step()

        # Clamp to epsilon bounds and apply constraints
        with torch.no_grad():
            p.data.clamp_(min=-epsilon_tensor, max=epsilon_tensor)
            p.data = pert_constraints(p.data, feature_list, final_step)

        # Zero gradients after constraints
        p.grad.zero_()

        if step % 10 == 0:
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                accuracy = (pred == labels).float().mean()
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}, "
                      f"Cost: {cost.item():.4f}, Classification Loss: {classification_loss.item():.4f}, "
                      f"Accuracy: {accuracy.item():.4f}")

    return p.detach()

def calculate_comprehensive_metrics(y_true, y_pred_original, y_pred_perturbed):
    """Calculate comprehensive evaluation metrics"""
    y_true = np.array(y_true)
    y_pred_original = np.array(y_pred_original)
    y_pred_perturbed = np.array(y_pred_perturbed)

    # Original model performance
    original_accuracy = accuracy_score(y_true, y_pred_original)
    original_precision = precision_score(y_true, y_pred_original, average='binary', zero_division=0)
    original_recall = recall_score(y_true, y_pred_original, average='binary', zero_division=0)
    original_f1 = f1_score(y_true, y_pred_original, average='binary', zero_division=0)

    # Perturbed model performance
    perturbed_accuracy = accuracy_score(y_true, y_pred_perturbed)
    perturbed_precision = precision_score(y_true, y_pred_perturbed, average='binary', zero_division=0)
    perturbed_recall = recall_score(y_true, y_pred_perturbed, average='binary', zero_division=0)
    perturbed_f1 = f1_score(y_true, y_pred_perturbed, average='binary', zero_division=0)

    # Attack success rate
    attack_success_rate = (y_pred_original != y_pred_perturbed).mean()

    metrics = {
        'original_accuracy': original_accuracy,
        'original_precision': original_precision,
        'original_recall': original_recall,
        'original_f1': original_f1,
        'perturbed_accuracy': perturbed_accuracy,
        'perturbed_precision': perturbed_precision,
        'perturbed_recall': perturbed_recall,
        'perturbed_f1': perturbed_f1,
        'attack_success_rate': attack_success_rate,
        'accuracy_drop': original_accuracy - perturbed_accuracy,
        'precision_drop': original_precision - perturbed_precision,
        'recall_drop': original_recall - perturbed_recall,
        'f1_drop': original_f1 - perturbed_f1
    }

    return metrics

def calculate_perturbation_cost(perturbation, feature_list, model_type, cost_type='DC'):
    """Calculate the cost of perturbation"""
    dummy_original = torch.zeros_like(perturbation).unsqueeze(0)
    dummy_perturbed = perturbation.unsqueeze(0)

    yopo_cost = compute_cost_graph(dummy_perturbed, dummy_original, feature_list, model_type, cost_type).item()

    l1_cost = torch.norm(perturbation, p=1).item()
    l2_cost = torch.norm(perturbation, p=2).item()
    total_features_changed = (torch.abs(perturbation) > 1e-6).sum().item()

    cost_metrics = {
        'total_cost': yopo_cost,
        'l1_cost': l1_cost,
        'l2_cost': l2_cost,
        'features_changed': total_features_changed,
        'cost_type': cost_type,
        'max_perturbation': torch.max(torch.abs(perturbation)).item(),
        'mean_perturbation': torch.mean(torch.abs(perturbation)).item()
    }

    return cost_metrics

def evaluate_attack_sklearn(sklearn_model, original_data, perturbed_data, feature_columns):
    """Evaluate attack success rate against sklearn model"""
    original_df = pd.DataFrame(original_data, columns=feature_columns)
    perturbed_df = pd.DataFrame(perturbed_data, columns=feature_columns)

    # Get predictions
    original_pred = sklearn_model.predict(original_df)
    perturbed_pred = sklearn_model.predict(perturbed_df)

    # Calculate attack success rate
    attack_success = (original_pred != perturbed_pred).mean()

    return attack_success, original_pred, perturbed_pred

def main():
    parser = argparse.ArgumentParser(description='YOPO Attack on AdGraph/WebGraph Models')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['adgraph', 'webgraph'],
                       help='Model type to attack')
    parser.add_argument('--data-path', type=str,
                       default='../../dataset/testing',
                       help='Directory containing target datasets')
    parser.add_argument('--cost-type', type=str, default='DC',
                       choices=['DC', 'HSC', 'HCC'],
                       help='Cost model type (DC, HSC, HCC)')
    parser.add_argument('--epsilon', type=int, default=10,
                       help='Maximum perturbation magnitude')
    parser.add_argument('--lagrangian', type=float, default=400,
                       help='Lagrangian multiplier for cost constraint')
    parser.add_argument('--num-steps', type=int, default=100,
                       help='Number of optimization steps')
    parser.add_argument('--step-size', type=float, default=0.1,
                       help='Optimization step size')
    parser.add_argument('--query-size', type=int, default=100000,
                       help='Query set size (original YOPO uses 100k)')
    parser.add_argument('--sampling-size', type=int, default=40000,
                       help='Number of samples for attack (sampled from query set)')
    parser.add_argument('--num-iterations', type=int, default=10,
                       help='Number of attack iterations to run for statistical analysis')
    parser.add_argument('--output-dir', type=str, default='graph_attack_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and features based on type
    if args.model_type == 'adgraph':
        model_path = ADGRAPH_MODEL_PATH
        print(f"Loading AdGraph model from {model_path}")
    else:
        model_path = WEBGRAPH_MODEL_PATH
        print(f"Loading WebGraph model from {model_path}")
    features_path=FEATURES_PATH

    # Load feature list
    feature_list = load_feature_list(features_path, args.model_type)
    print(f"Loaded {len(feature_list)} features for {args.model_type}")
    print(f"Features: {feature_list}")

    # Load sklearn model with compatibility handling
    sklearn_model = load_sklearn_model_compat(model_path)
    if sklearn_model is not None:
        print(f"Loaded sklearn model: {type(sklearn_model)}")
    else:
        print(f"Sklearn model loading skipped due to version incompatibility")
        print(f"Will only evaluate surrogate model attacks")

    # Load data
    dataset = GraphDataset(args.data_path, feature_list)

    # Two-stage sampling following original YOPO
    # Stage 1: Create query set (100k samples like original YOPO)
    dataset_size = len(dataset)
    query_size = min(args.query_size, dataset_size)
    print(f"\nTwo-stage sampling (matching original YOPO):")
    print(f"  Full dataset size: {dataset_size}")
    print(f"  Query set size: {query_size}")

    # Stage 2: Sample from query set
    # Note: If sampling_size > query_size, we'll use replacement in DataLoader
    train_size = min(args.sampling_size, query_size)
    query_dataset, _ = random_split(dataset, [query_size, dataset_size - query_size])

    # If we need more samples than query_size, we sample with replacement via shuffling
    # This matches original YOPO behavior where sampling_size (40k) can be > query_size
    if args.sampling_size > query_size:
        print(f"  Sampling size ({args.sampling_size}) > query size ({query_size})")
        print(f"  Will sample {args.sampling_size} with replacement in each iteration")
        train_size = args.sampling_size
        # Use replacement sampling by creating a custom sampler
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(query_dataset, replacement=True, num_samples=train_size)
        train_dataloader = DataLoader(query_dataset, batch_size=train_size, sampler=sampler)
    else:
        print(f"  Sampling {args.sampling_size} from query set (no replacement needed)")
        train_dataset, _ = random_split(query_dataset, [train_size, query_size - train_size])
        train_dataloader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize surrogate model
    model = SurrogateModel(len(feature_list)).to(device)

    # Train surrogate model (matching original YOPO: 30 epochs, lr=5e-4)
    # CRITICAL: Train on sklearn model predictions, not ground truth!
    print("Training surrogate model...")

    if sklearn_model is not None:
        print("  Generating sklearn predictions for surrogate training...")
        # Collect all training features first
        all_features = []
        for batch_features, _ in train_dataloader:
            all_features.append(batch_features)
        train_features = torch.cat(all_features, dim=0)

        # Get sklearn predictions for all training data
        train_np = train_features.numpy()
        train_df = pd.DataFrame(train_np, columns=feature_list)
        sklearn_predictions = sklearn_model.predict(train_df)
        sklearn_labels = torch.tensor(sklearn_predictions, dtype=torch.long)

        # Create 90/10 train/test split (matching original YOPO)
        from torch.utils.data import TensorDataset
        full_dataset = TensorDataset(train_features, sklearn_labels)
        dataset_size = len(full_dataset)
        train_split_size = int(dataset_size * 0.9)
        test_split_size = dataset_size - train_split_size

        surrogate_train_dataset, surrogate_test_dataset = random_split(
            full_dataset, [train_split_size, test_split_size]
        )

        # Use batch_size=256 (compromise between original batch_size=30 and full batch)
        train_dataloader = DataLoader(surrogate_train_dataset, batch_size=256, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(surrogate_test_dataset, batch_size=256, shuffle=False)

        print(f"  Training on {len(surrogate_train_dataset)} samples, validating on {len(surrogate_test_dataset)}")
        print(f"  Sklearn prediction distribution: {np.bincount(sklearn_predictions)}")

        # Store full dataset for attack
        surrogate_dataset = full_dataset
    else:
        print("  WARNING: Training on ground truth labels (sklearn model not available)")
        test_dataloader = None

    # Enhanced training configuration for WebGraph
    num_epochs = 50 if args.model_type == 'webgraph' else 30
    learning_rate = 5e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"  Training with {num_epochs} epochs, batch_size=256, lr={learning_rate}")
    best_test_acc = 0.0

    model.train()
    for epoch in range(num_epochs):
        # Training phase
        correct = 0
        total = 0
        epoch_loss = 0
        for batch_features, batch_labels in train_dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # Count correct predictions
            predictions = torch.argmax(outputs, dim=1)
            correct += torch.sum(predictions == batch_labels).item()
            total += batch_labels.size(0)
            epoch_loss += loss.item()

        train_accuracy = 100 * correct / total

        # Validation phase (if test set available)
        test_accuracy = 0.0
        if test_dataloader is not None:
            model.eval()
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for batch_features, batch_labels in test_dataloader:
                    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                    outputs = model(batch_features)
                    pred = torch.argmax(outputs, dim=1)
                    test_correct += (pred == batch_labels).sum().item()
                    test_total += batch_labels.size(0)
                test_accuracy = 100 * test_correct / test_total
                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy
            model.train()

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            if test_dataloader is not None:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}% (Best: {best_test_acc:.2f}%)")
            else:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Accuracy: {train_accuracy:.2f}%")

    if test_dataloader is not None:
        print(f"Surrogate model training completed (Best test accuracy: {best_test_acc:.2f}%)")
    else:
        print(f"Surrogate model training completed (Train accuracy: {train_accuracy:.2f}%)")

    # Recreate dataloader with full batch size for attack
    if sklearn_model is not None:
        attack_dataloader = DataLoader(surrogate_dataset, batch_size=train_size, shuffle=True)
    else:
        attack_dataloader = train_dataloader

    # Storage for iteration results
    iteration_results = {
        'surrogate_asr': [],
        'surrogate_accuracy_drop': [],
        'surrogate_f1_drop': [],
        'surrogate_precision_drop': [],
        'surrogate_recall_drop': [],
        'sklearn_asr': [],
        'sklearn_accuracy_drop': [],
        'sklearn_f1_drop': [],
        'sklearn_precision_drop': [],
        'sklearn_recall_drop': [],
        'total_cost': [],
        'features_changed': []
    }

    # Run multiple iterations
    print(f"\n{'='*60}")
    print(f"Starting {args.num_iterations} attack iterations")
    print(f"{'='*60}\n")

    for iteration in range(args.num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{args.num_iterations}")
        print(f"{'='*60}\n")

        # Perform UAP attack
        for batch_features, batch_labels in attack_dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            perturbation = uap_attack(
                model=model,
                features=batch_features,
                labels=batch_labels,
                feature_list=feature_list,
                model_type=args.model_type,
                epsilon=args.epsilon,
                step_size=args.step_size,
                num_steps=args.num_steps,
                lagrangian=args.lagrangian,
                cost_type=args.cost_type,
                device=device
            )

            # Apply perturbation
            perturbed_features = batch_features + perturbation.unsqueeze(0).expand_as(batch_features)

            # Evaluate surrogate model
            with torch.no_grad():
                original_pred = torch.argmax(model(batch_features), dim=1)
                perturbed_pred = torch.argmax(model(perturbed_features), dim=1)
                surrogate_asr = (original_pred != perturbed_pred).float().mean()
                print(f"Iteration {iteration + 1} - Surrogate ASR: {surrogate_asr.item():.4f}")

                surrogate_metrics = calculate_comprehensive_metrics(
                    batch_labels.cpu().numpy(),
                    original_pred.cpu().numpy(),
                    perturbed_pred.cpu().numpy()
                )

                cost_metrics = calculate_perturbation_cost(perturbation, feature_list, args.model_type, args.cost_type)

                print(f"  Original Accuracy: {surrogate_metrics['original_accuracy']:.4f}")
                print(f"  Perturbed Accuracy: {surrogate_metrics['perturbed_accuracy']:.4f}")
                print(f"  Accuracy Drop: {surrogate_metrics['accuracy_drop']:.4f}")
                print(f"  F1 Drop: {surrogate_metrics['f1_drop']:.4f}")
                print(f"  Total Cost ({args.cost_type}): {cost_metrics['total_cost']:.4f}")
                print(f"  Features Changed: {cost_metrics['features_changed']}")

                # Store surrogate results
                iteration_results['surrogate_asr'].append(surrogate_asr.item())
                iteration_results['surrogate_accuracy_drop'].append(surrogate_metrics['accuracy_drop'])
                iteration_results['surrogate_f1_drop'].append(surrogate_metrics['f1_drop'])
                iteration_results['surrogate_precision_drop'].append(surrogate_metrics['precision_drop'])
                iteration_results['surrogate_recall_drop'].append(surrogate_metrics['recall_drop'])
                iteration_results['total_cost'].append(cost_metrics['total_cost'])
                iteration_results['features_changed'].append(cost_metrics['features_changed'])

            # Save perturbation
            perturbation_file = os.path.join(args.output_dir,
                                           f"perturbation_{args.model_type}_eps{args.epsilon}_{args.cost_type}_{args.sampling_size}_iter{iteration+1}.npy")
            np.save(perturbation_file, perturbation.cpu().numpy())

            # Evaluate against sklearn model (if available)
            if sklearn_model is not None:
                try:
                    original_data = batch_features.cpu().numpy()
                    perturbed_data = perturbed_features.cpu().numpy()

                    sklearn_asr, orig_pred, pert_pred = evaluate_attack_sklearn(
                        sklearn_model, original_data, perturbed_data, feature_list
                    )

                    print(f"  Sklearn Model ASR: {sklearn_asr:.4f}")

                    sklearn_metrics = calculate_comprehensive_metrics(
                        batch_labels.cpu().numpy(),
                        orig_pred,
                        pert_pred
                    )

                    print(f"  Sklearn Original Accuracy: {sklearn_metrics['original_accuracy']:.4f}")
                    print(f"  Sklearn Perturbed Accuracy: {sklearn_metrics['perturbed_accuracy']:.4f}")
                    print(f"  Sklearn Accuracy Drop: {sklearn_metrics['accuracy_drop']:.4f}")
                    print(f"  Sklearn F1 Drop: {sklearn_metrics['f1_drop']:.4f}")

                    # Store sklearn results
                    iteration_results['sklearn_asr'].append(sklearn_asr)
                    iteration_results['sklearn_accuracy_drop'].append(sklearn_metrics['accuracy_drop'])
                    iteration_results['sklearn_f1_drop'].append(sklearn_metrics['f1_drop'])
                    iteration_results['sklearn_precision_drop'].append(sklearn_metrics['precision_drop'])
                    iteration_results['sklearn_recall_drop'].append(sklearn_metrics['recall_drop'])

                except Exception as e:
                    print(f"  Could not evaluate sklearn model: {e}")
            else:
                print(f"  Sklearn model evaluation skipped (not loaded)")

            break  # Only process first batch

    # Calculate and display statistics
    import json
    print(f"\n{'='*60}")
    print(f"STATISTICAL SUMMARY ({args.num_iterations} iterations)")
    print(f"{'='*60}\n")

    def calc_stats(values):
        if not values:
            return {}
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    statistics = {
        'configuration': {
            'model_type': args.model_type,
            'cost_type': args.cost_type,
            'epsilon': args.epsilon,
            'lagrangian': args.lagrangian,
            'num_steps': args.num_steps,
            'step_size': args.step_size,
            'query_size': query_size,
            'sampling_size': args.sampling_size,
            'num_iterations': args.num_iterations,
            'total_features': len(feature_list)
        },
        'surrogate_model': {
            'asr': calc_stats(iteration_results['surrogate_asr']),
            'accuracy_drop': calc_stats(iteration_results['surrogate_accuracy_drop']),
            'f1_drop': calc_stats(iteration_results['surrogate_f1_drop']),
            'precision_drop': calc_stats(iteration_results['surrogate_precision_drop']),
            'recall_drop': calc_stats(iteration_results['surrogate_recall_drop'])
        },
        'cost': {
            'total_cost': calc_stats(iteration_results['total_cost']),
            'features_changed': calc_stats(iteration_results['features_changed'])
        }
    }

    # Add sklearn statistics
    if iteration_results['sklearn_asr']:
        statistics['sklearn_model'] = {
            'asr': calc_stats(iteration_results['sklearn_asr']),
            'accuracy_drop': calc_stats(iteration_results['sklearn_accuracy_drop']),
            'f1_drop': calc_stats(iteration_results['sklearn_f1_drop']),
            'precision_drop': calc_stats(iteration_results['sklearn_precision_drop']),
            'recall_drop': calc_stats(iteration_results['sklearn_recall_drop'])
        }

    # Display statistics
    print("SURROGATE MODEL:")
    print(f"  ASR:          mean={statistics['surrogate_model']['asr']['mean']:.4f}, "
          f"median={statistics['surrogate_model']['asr']['median']:.4f}, "
          f"std={statistics['surrogate_model']['asr']['std']:.4f}")
    print(f"  Accuracy Drop: mean={statistics['surrogate_model']['accuracy_drop']['mean']:.4f}, "
          f"median={statistics['surrogate_model']['accuracy_drop']['median']:.4f}, "
          f"std={statistics['surrogate_model']['accuracy_drop']['std']:.4f}")
    print(f"  F1 Drop:      mean={statistics['surrogate_model']['f1_drop']['mean']:.4f}, "
          f"median={statistics['surrogate_model']['f1_drop']['median']:.4f}, "
          f"std={statistics['surrogate_model']['f1_drop']['std']:.4f}")

    if 'sklearn_model' in statistics:
        print(f"\n{args.model_type.upper()} MODEL:")
        print(f"  ASR:          mean={statistics['sklearn_model']['asr']['mean']:.4f}, "
              f"median={statistics['sklearn_model']['asr']['median']:.4f}, "
              f"std={statistics['sklearn_model']['asr']['std']:.4f}")
        print(f"  Accuracy Drop: mean={statistics['sklearn_model']['accuracy_drop']['mean']:.4f}, "
              f"median={statistics['sklearn_model']['accuracy_drop']['median']:.4f}, "
              f"std={statistics['sklearn_model']['accuracy_drop']['std']:.4f}")
        print(f"  F1 Drop:      mean={statistics['sklearn_model']['f1_drop']['mean']:.4f}, "
              f"median={statistics['sklearn_model']['f1_drop']['median']:.4f}, "
              f"std={statistics['sklearn_model']['f1_drop']['std']:.4f}")

    print(f"\nCOST:")
    print(f"  Total Cost:   mean={statistics['cost']['total_cost']['mean']:.4f}, "
          f"median={statistics['cost']['total_cost']['median']:.4f}, "
          f"std={statistics['cost']['total_cost']['std']:.4f}")
    print(f"  Features Changed: mean={statistics['cost']['features_changed']['mean']:.1f}, "
          f"median={statistics['cost']['features_changed']['median']:.1f}")

    # Save statistics
    stats_file = os.path.join(args.output_dir,
                             f"statistics_{args.model_type}_eps{args.epsilon}_{args.cost_type}_{args.sampling_size}.json")

    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj

    statistics = convert_to_native(statistics)
    statistics['raw_results'] = convert_to_native(iteration_results)

    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Statistics saved to: {stats_file}")
    print(f"{'='*60}")

    print("\nAttack completed!")

if __name__ == "__main__":
    main()
