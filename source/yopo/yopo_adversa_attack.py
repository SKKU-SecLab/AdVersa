"""
YOPO Attack Script for AdVersa Models
Adapted from the original YOPO framework to work with pre-trained AdVersa models.

This script supports two attack modes:
1. Constrained: Prevents changes to codemelt features
2. Unconstrained: Allows changes to all features including codemelt features
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import joblib
import argparse
import os
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Feature mapping for AdVersa models
ADVERSA_MODELS = ['AdFlush_reimpl','AdVersa_consistentimportance','AdVersa_frequencyfiltering','AdVersa_intersection','AdVersa_union']

class AdVersaDataset(Dataset):
    """Custom Dataset for AdVersa data"""
    def __init__(self, data_file, feature_columns=None):
        # Load parquet or CSV file
        if data_file.endswith('.parquet'):
            self.data = pd.read_parquet(data_file)
        else:
            self.data = pd.read_csv(data_file)
            
        if feature_columns is not None:
            # Filter to only include specified features
            available_features = [f for f in feature_columns if f in self.data.columns]
            label_col = 'label' if 'label' in self.data.columns else 'CLASS'
            self.data = self.data[available_features + [label_col] if label_col in self.data.columns else available_features]
        
        # Handle label column (could be 'label' or 'CLASS')
        label_col = None
        if 'label' in self.data.columns:
            label_col = 'label'
        elif 'CLASS' in self.data.columns:
            label_col = 'CLASS'
            
        if label_col is not None:
            self.features = self.data.drop(label_col, axis=1).values.astype(np.float32)
            self.labels = self.data[label_col].values.astype(np.long)
        else:
            self.features = self.data.values.astype(np.float32)
            self.labels = np.zeros(len(self.features), dtype=np.long)  # Dummy labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

class SurrogateModel(nn.Module):
    """Surrogate neural network model"""
    def __init__(self, input_size):
        super(SurrogateModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        return x

def load_feature_set(model_name):
    """Load feature set for a specific AdVersa model"""
    if model_name not in ADVERSA_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {ADVERSA_MODELS}")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    features=config["model_"+model_name]
    features = np.load(feature_path, allow_pickle=True)
    return features

def identify_codemelt_features(feature_list):
    """Identify codemelt features in the feature list"""
    return [f for f in feature_list if f.startswith('codemalt_')]

def create_constraint_mask(feature_list, constrained_mode=True):
    """Create a mask for feature constraints"""
    if not constrained_mode:
        # Unconstrained: allow changes to all features
        return np.ones(len(feature_list), dtype=bool)
    
    # Constrained: prevent changes to codemelt features
    codemelt_features = identify_codemelt_features(feature_list)
    mask = np.ones(len(feature_list), dtype=bool)
    
    for i, feature in enumerate(feature_list):
        if feature in codemelt_features:
            mask[i] = False
    
    return mask

def inject_feature(features, perturbation, feature_list, constrained_mode=True):
    """Inject perturbation into features following original YOPO logic"""
    result = features.clone()
    
    # Get perturbable feature indices
    constraint_mask = create_constraint_mask(feature_list, constrained_mode)
    
    # Apply perturbation only to allowed features
    for i, allowed in enumerate(constraint_mask):
        if allowed:
            result[:, i] += perturbation[:, i]  # Use broadcasting properly
    
    return result

def compute_cost_adflush(injected_features, orig_features, feature_list, cost_type='DC'):
    """Compute cost following original YOPO cost model for AdFlush with exact YOPO values"""
    cost = torch.tensor(0.0, device=injected_features.device)
    diff_features = injected_features - orig_features
    
    # Original YOPO cost values from cost_dict_adflush.py
    if cost_type == 'DC':
        STRUCT_WEIGHT = 1
        CONTENT_WEIGHT = 1
        JAVASCRIPT_WEIGHT = 1
    elif cost_type == 'HJC':
        STRUCT_WEIGHT = 1
        CONTENT_WEIGHT = 1
        JAVASCRIPT_WEIGHT = 10
    elif cost_type == 'HCC':
        STRUCT_WEIGHT = 1
        CONTENT_WEIGHT = 10
        JAVASCRIPT_WEIGHT = 1
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")
    
    # Base cost constants
    COST_STRUCT_EASY_EASY = 0.1 * STRUCT_WEIGHT
    COST_STRUCT_EASY = 1 * STRUCT_WEIGHT
    COST_STRUCT_MID_EASY = 2 * STRUCT_WEIGHT
    COST_STRUCT_MID = 2 * STRUCT_WEIGHT
    COST_STRUCT_HARD = 3 * STRUCT_WEIGHT
    
    COST_CONTENT_EASY_EASY = 0.2 * CONTENT_WEIGHT
    COST_CONTENT_EASY = 1 * CONTENT_WEIGHT
    COST_CONTENT_MID_EASY = 0.2 * CONTENT_WEIGHT
    COST_CONTENT_MID = 2 * CONTENT_WEIGHT
    COST_CONTENT_HARD = 3 * CONTENT_WEIGHT
    
    COST_JS_EASY = 1 * JAVASCRIPT_WEIGHT
    COST_JS_MID = 2 * JAVASCRIPT_WEIGHT
    COST_JS_HARD = 3 * JAVASCRIPT_WEIGHT
    
    COST_FLOW_EASY = 1
    COST_FLOW_MID = 2
    COST_FLOW_HARD = 2
    
    # Original YOPO feature costs plus AdVersa mappings
    feature_costs = {
        "num_nodes": COST_STRUCT_EASY_EASY,
        "num_edges": COST_STRUCT_EASY_EASY,
        "in_out_degree": COST_STRUCT_EASY_EASY,
        "average_degree_connectivity": COST_STRUCT_HARD,
        "ascendant_has_ad_keyword_0": COST_STRUCT_HARD / 2,
        "ascendant_has_ad_keyword_1": COST_STRUCT_HARD / 2,
        
        "url_length": COST_CONTENT_MID_EASY,
        "keyword_raw_present_0": COST_CONTENT_HARD / 2,
        "keyword_raw_present_1": COST_CONTENT_HARD / 2,
        "is_third_party_0": COST_CONTENT_HARD / 2,
        "is_third_party_1": COST_CONTENT_HARD / 2,
        "is_third_party": COST_CONTENT_HARD / 2,  # AdVersa version
        
        "brackettodot": COST_JS_EASY,
        "num_get_storage": COST_FLOW_MID,
        "num_set_storage": COST_FLOW_MID,
        "num_get_cookie": COST_FLOW_MID,
        "num_set_cookie": COST_FLOW_MID,  # AdVersa version
        "num_requests_sent": COST_FLOW_EASY,
        "avg_ident": COST_JS_MID,
        "avg_charperline": COST_JS_MID,
        "ast_depth": COST_JS_MID,  # AdVersa feature
        
        "ng_0_0_2": COST_JS_EASY,
        "ng_0_15_15": COST_JS_EASY,
        "ng_2_13_2": COST_JS_EASY,
        "ng_15_0_3": COST_JS_EASY,
        "ng_15_0_15": COST_JS_EASY,
        "ng_15_15_15": COST_JS_EASY,
        
        # AdVersa specific features
        "content_policy_type": COST_CONTENT_MID_EASY,
        "keyword_char_present": COST_CONTENT_HARD / 2,
        "ad_size_in_qs_present": COST_CONTENT_MID_EASY,
    }
    
    # Calculate weighted cost
    for i, feature_name in enumerate(feature_list):
        if feature_name in feature_costs:
            weight = feature_costs[feature_name]
        elif feature_name.startswith('url_nomic_') or feature_name.startswith('fqdn_nomic_'):
            # URL/FQDN nomic embeddings - treat as content features
            # weight = COST_CONTENT_MID_EASY
            weight = COST_CONTENT_MID_EASY
        elif feature_name.startswith('codemalt_'):
            # CodeMelt features - treat as JavaScript features
            weight = COST_JS_HARD
        else:
            # Default cost for other unknown features - treat as content
            weight = COST_CONTENT_EASY
        
        cost += weight * diff_features[:, i].abs().sum()
    
    return cost / diff_features.shape[0]  # Divide by batch size like original YOPO

def pert_constraints(pert, feature_list, constrained_mode=True, final_step=False):
    """Apply perturbation constraints following original YOPO logic"""
    # Create constraint mask
    constraint_mask = torch.tensor(
        create_constraint_mask(feature_list, constrained_mode),
        dtype=torch.float32,
        device=pert.device
    )
    
    # Zero out non-perturbable features
    pert = pert * constraint_mask
    
    # Additional constraints for final step (could add categorical/binary constraints here)
    if final_step:
        # For now, just ensure constraint mask is applied
        pert = pert * constraint_mask
    
    return pert

def uap_attack(model, features, labels, feature_list, epsilon=10, step_size=0.1, 
               num_steps=100, lagrangian=400, constrained_mode=True, cost_type='DC', device='cpu'):
    """
    Universal Adversarial Perturbation attack following original YOPO implementation
    
    Args:
        model: Trained surrogate model
        features: Input features tensor
        labels: Ground truth labels tensor  
        feature_list: List of feature names
        epsilon: Maximum perturbation magnitude
        step_size: Step size for optimization
        num_steps: Number of optimization steps
        lagrangian: Lagrangian multiplier for cost constraint
        constrained_mode: If True, prevent codemelt feature changes
        cost_type: Cost model type ('DC', 'HJC', 'HCC')
        device: Device to run on
    """
    
    # Initialize perturbation (following original YOPO)
    p = torch.zeros_like(features[0]).to(device)
    p.requires_grad_(True)
    
    features = features.to(device)
    labels = labels.to(device)
    train_size = features.shape[0]
    
    print(f"Starting UAP attack (constrained_mode={constrained_mode})")
    print(f"Codemelt features: {len([f for f in feature_list if f.startswith('codemalt_')])}")
    constraint_mask = create_constraint_mask(feature_list, constrained_mode)
    print(f"Perturbable features: {sum(constraint_mask)}/{len(feature_list)}")
    
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
    print(f"  Sample epsilon bounds: {epsilon_tensor[:5].tolist()}")
    
    # Scale step size with epsilon to encourage larger perturbations
    scaled_step_size = step_size * (epsilon / 10.0)  # Scale relative to epsilon=10 baseline
    print(f"  Scaled step size: {scaled_step_size:.4f} (original: {step_size})")
    
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    
    for step in range(num_steps):
        final_step = (step == num_steps - 1)
        
        # Expand perturbation to match batch size (following original YOPO)
        p_exp = p.repeat(train_size, 1)
        
        # Inject perturbation into features
        injected_features = inject_feature(features, p_exp, feature_list, constrained_mode)
        
        # Forward pass
        outputs = model(injected_features)
        
        # Compute cost (following original YOPO cost computation)
        cost = compute_cost_adflush(injected_features, features, feature_list, cost_type)
        
        # Calculate loss (following original YOPO: cost - lagrangian * classification_loss)
        classification_loss = loss_fn(outputs, labels)
        loss = cost - (lagrangian * classification_loss)
        
        # Zero gradients and backward pass
        if p.grad is not None:
            p.grad.zero_()
        loss.backward()
        
        # Manual gradient update (following original YOPO)
        with torch.no_grad():
            if p.grad is not None:
                # Sign of gradient (following original YOPO)
                grad_sign = p.grad.sign()
                
                # Update perturbation
                p.data.sub_(grad_sign * scaled_step_size)
                
                # Clamp to epsilon bounds
                p_before = p.data.clone()
                p.data.clamp_(min=-epsilon_tensor, max=epsilon_tensor)
                
                if step % 50 == 0:  # Debug output every 50 steps
                    print(f"  Step {step} clamp debug:")
                    print(f"    P before clamp min/max: {p_before.min().item():.3f}/{p_before.max().item():.3f}")
                    print(f"    P after clamp min/max: {p.data.min().item():.3f}/{p.data.max().item():.3f}")
                    print(f"    Epsilon bounds min/max: {(-epsilon_tensor).min().item():.3f}/{epsilon_tensor.max().item():.3f}")
                
                # Apply constraints
                p = pert_constraints(p, feature_list, constrained_mode, final_step)
        
        if step % 10 == 0:
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                accuracy = (pred == labels).float().mean()
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}, "
                      f"Cost: {cost.item():.4f}, Classification Loss: {classification_loss.item():.4f}, "
                      f"Accuracy: {accuracy.item():.4f}")
    
    return p.detach()

def load_h2o_model(model_path):
    """Load H2O AutoML model"""
    h2o.init()
    # return h2o.load_model(model_path)
    return h2o.import_mojo(model_path)

def calculate_comprehensive_metrics(y_true, y_pred_original, y_pred_perturbed):
    """Calculate comprehensive evaluation metrics"""
    
    # Convert to numpy arrays if they aren't already
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
    
    # Attack success rate (prediction flips)
    attack_success_rate = (y_pred_original != y_pred_perturbed).mean()
    
    # Confusion matrices
    original_cm = confusion_matrix(y_true, y_pred_original)
    perturbed_cm = confusion_matrix(y_true, y_pred_perturbed)
    
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
        'f1_drop': original_f1 - perturbed_f1,
        'original_confusion_matrix': original_cm.tolist(),
        'perturbed_confusion_matrix': perturbed_cm.tolist()
    }
    
    return metrics

def calculate_perturbation_cost(perturbation, feature_list, cost_type='DC'):
    """Calculate the cost of perturbation based on cost model using YOPO approach"""
    
    # Create dummy features to use the proper cost computation
    dummy_original = torch.zeros_like(perturbation).unsqueeze(0)
    dummy_perturbed = perturbation.unsqueeze(0)
    
    # Use the proper YOPO cost computation
    yopo_cost = compute_cost_adflush(dummy_perturbed, dummy_original, feature_list, cost_type).item()
    
    # Also calculate traditional metrics for comparison
    l1_cost = torch.norm(perturbation, p=1).item()
    l2_cost = torch.norm(perturbation, p=2).item()
    total_features_changed = (torch.abs(perturbation) > 1e-6).sum().item()
    
    cost_metrics = {
        'total_cost': yopo_cost,  # Use YOPO cost as the main cost
        'l1_cost': l1_cost,
        'l2_cost': l2_cost,
        'features_changed': total_features_changed,
        'cost_type': cost_type,
        'max_perturbation': torch.max(torch.abs(perturbation)).item(),
        'mean_perturbation': torch.mean(torch.abs(perturbation)).item()
    }
    
    return cost_metrics

def evaluate_attack_h2o(h2o_model, original_data, perturbed_data, feature_columns):
    """Evaluate attack success rate against H2O model"""
    
    # Convert to H2O frames
    original_h2o = h2o.H2OFrame(pd.DataFrame(original_data, columns=feature_columns))
    perturbed_h2o = h2o.H2OFrame(pd.DataFrame(perturbed_data, columns=feature_columns))
    
    # Get predictions
    original_pred = h2o_model.predict(original_h2o).as_data_frame()['predict'].values
    perturbed_pred = h2o_model.predict(perturbed_h2o).as_data_frame()['predict'].values
    
    # Calculate attack success rate (prediction flips)
    attack_success = (original_pred != perturbed_pred).mean()
    
    return attack_success, original_pred, perturbed_pred

def main():
    parser = argparse.ArgumentParser(description='YOPO Attack on AdVersa Models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=ADVERSA_MODELS,
                       help='AdVersa model variant to attack')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--h2o-model-path', type=str, 
                       help='Path to H2O model file (optional)')
    parser.add_argument('--constrained', action='store_true',
                       help='Use constrained mode (prevent codemelt changes)')
    parser.add_argument('--cost-type', type=str, default='DC', 
                       choices=['DC', 'HJC', 'HCC'],
                       help='Cost model type (DC, HJC, HCC)')
    parser.add_argument('--epsilon', type=int, default=10,
                       help='Maximum perturbation magnitude')
    parser.add_argument('--lagrangian', type=float, default=400,
                       help='Lagrangian multiplier for cost constraint')
    parser.add_argument('--num-steps', type=int, default=100,
                       help='Number of optimization steps')
    parser.add_argument('--step-size', type=float, default=0.1,
                       help='Optimization step size')
    parser.add_argument('--sampling-size', type=int, default=20000,
                       help='Number of samples for attack (recommended: 20k, 40k, 60k)')
    parser.add_argument('--output-dir', type=str, default='attack_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load feature set for the model
    feature_list = load_feature_set(args.model)
    print(f"Loaded feature set for {args.model}: {len(feature_list)} features")
    
    # Identify codemelt features
    codemelt_features = identify_codemelt_features(feature_list)
    print(f"Found {len(codemelt_features)} codemelt features")
    
    # Load data
    dataset = AdVersaDataset(args.data_path, feature_list)
    
    # Create data loader
    dataset_size = len(dataset)
    train_size = min(args.sampling_size, dataset_size)
    train_dataset, _ = random_split(dataset, [train_size, dataset_size - train_size])
    train_dataloader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize surrogate model
    model = SurrogateModel(len(feature_list)).to(device)
    
    # For now, we'll train a simple surrogate model
    # In practice, you would load a pre-trained surrogate model
    print("Training surrogate model...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(10):  # Quick training
        for batch_features, batch_labels in train_dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    print("Surrogate model training completed")
    
    # Evaluate model accuracy
    model.eval()
    with torch.no_grad():
        for batch_features, batch_labels in train_dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            pred = torch.argmax(outputs, dim=1)
            accuracy = (pred == batch_labels).float().mean()
            print(f"Surrogate model accuracy: {accuracy.item():.4f}")
    
    # Perform UAP attack
    for batch_features, batch_labels in train_dataloader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        perturbation = uap_attack(
            model=model,
            features=batch_features,
            labels=batch_labels,
            feature_list=feature_list,
            epsilon=args.epsilon,
            step_size=args.step_size,
            # num_steps=args.num_steps,
            num_steps=args.epsilon * 10,
            lagrangian=args.lagrangian,
            constrained_mode=args.constrained,
            cost_type=args.cost_type,
            device=device
        )
        
        # Apply perturbation
        perturbed_features = batch_features + perturbation.unsqueeze(0).expand_as(batch_features)
        
        # Evaluate surrogate model attack success
        with torch.no_grad():
            original_pred = torch.argmax(model(batch_features), dim=1)
            perturbed_pred = torch.argmax(model(perturbed_features), dim=1)
            surrogate_asr = (original_pred != perturbed_pred).float().mean()
            print(f"Surrogate Attack Success Rate: {surrogate_asr.item():.4f}")
            
            # Calculate comprehensive metrics for surrogate model
            surrogate_metrics = calculate_comprehensive_metrics(
                batch_labels.cpu().numpy(),
                original_pred.cpu().numpy(), 
                perturbed_pred.cpu().numpy()
            )
            
            # Calculate perturbation cost
            cost_metrics = calculate_perturbation_cost(perturbation, feature_list, args.cost_type)
            
            print(f"Original Accuracy: {surrogate_metrics['original_accuracy']:.4f}")
            print(f"Perturbed Accuracy: {surrogate_metrics['perturbed_accuracy']:.4f}")
            print(f"Original Precision: {surrogate_metrics['original_precision']:.4f}")
            print(f"Perturbed Precision: {surrogate_metrics['perturbed_precision']:.4f}")
            print(f"Original Recall: {surrogate_metrics['original_recall']:.4f}")
            print(f"Perturbed Recall: {surrogate_metrics['perturbed_recall']:.4f}")
            print(f"Total Cost ({args.cost_type}): {cost_metrics['total_cost']:.4f}")
            print(f"Features Changed: {cost_metrics['features_changed']}")
            print(f"Max Perturbation: {cost_metrics['max_perturbation']:.6f}")
            print(f"Mean Perturbation: {cost_metrics['mean_perturbation']:.6f}")
        
        # Save perturbation
        mode_str = "constrained" if args.constrained else "unconstrained"
        perturbation_file = os.path.join(args.output_dir, 
                                       f"perturbation_{args.model}_{mode_str}_eps{args.epsilon}_{args.cost_type}_{args.sampling_size}.npy")
        np.save(perturbation_file, perturbed_features.cpu().numpy())
        print(f"Saved perturbation to: {perturbation_file}")
        
        # Save results
        results = {
            'model': args.model,
            'constrained_mode': args.constrained,
            'cost_type': args.cost_type,
            'epsilon': args.epsilon,
            'codemelt_features_count': len(codemelt_features),
            'total_features': len(feature_list),
            'perturbable_features': len(feature_list) - (len(codemelt_features) if args.constrained else 0),
            
            # Surrogate model metrics
            'surrogate_asr': surrogate_asr.item(),
            'surrogate_metrics': surrogate_metrics,
            'cost_metrics': cost_metrics,
            
            # Attack configuration
            'lagrangian': args.lagrangian,
            'num_steps': args.epsilon * 10,
            'step_size': args.step_size,
            'sampling_size': args.sampling_size
        }
        
        results_file = os.path.join(args.output_dir, 
                                  f"results_{args.model}_{mode_str}_eps{args.epsilon}_{args.cost_type}_{args.sampling_size}.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {results_file}")
        
        # Load and evaluate against H2O model if available
        if args.h2o_model_path:
            try:
                h2o_model = load_h2o_model(args.h2o_model_path)
                
                original_data = batch_features.cpu().numpy()
                perturbed_data = perturbed_features.cpu().numpy()
                
                h2o_asr, orig_pred, pert_pred = evaluate_attack_h2o(
                    h2o_model, original_data, perturbed_data, feature_list
                )
                
                print(f"H2O Model Attack Success Rate: {h2o_asr:.4f}")
                
                # Calculate comprehensive metrics for H2O model
                h2o_metrics = calculate_comprehensive_metrics(
                    batch_labels.cpu().numpy(),
                    orig_pred,
                    pert_pred
                )
                
                print(f"H2O Original Accuracy: {h2o_metrics['original_accuracy']:.4f}")
                print(f"H2O Perturbed Accuracy: {h2o_metrics['perturbed_accuracy']:.4f}")
                print(f"H2O Original Precision: {h2o_metrics['original_precision']:.4f}")
                print(f"H2O Perturbed Precision: {h2o_metrics['perturbed_precision']:.4f}")
                print(f"H2O Original Recall: {h2o_metrics['original_recall']:.4f}")
                print(f"H2O Perturbed Recall: {h2o_metrics['perturbed_recall']:.4f}")
                
                # Update results
                results['h2o_asr'] = h2o_asr
                results['h2o_metrics'] = h2o_metrics
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                print(f"Could not evaluate H2O model: {e}")
        else:
            print("No H2O model path provided, skipping H2O evaluation")
        
        break  # Only process first batch
    
    print("Attack completed!")

if __name__ == "__main__":
    main()