import os
import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

# Import custom modules
from hierarchical_model import HierarchicalRespiratoryClassifier, train_hierarchical_model
from data_preprocessing import SPRSoundDataset
from config import Config

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="SPRSound Hierarchical Respiratory Sound Classification")
    
    # Dataset parameters
    parser.add_argument("--train_wav", type=str, default="./train_wav", 
                        help="Directory with training WAV files")
    parser.add_argument("--train_json", type=str, default="./train_json", 
                        help="Directory with training JSON annotations")
    parser.add_argument("--test_wav", type=str, default="./test_wav", 
                        help="Directory with test WAV files")
    parser.add_argument("--test_json", type=str, default="./test_json", 
                        help="Directory with test JSON annotations")
    parser.add_argument("--include_2023", action="store_true", 
                        help="Include the 2023 dataset files")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=15, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, 
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, 
                        help="Early stopping patience")
    
    # Model parameters
    parser.add_argument("--model_dir", type=str, default="./hierarchical_model", 
                        help="Directory to save model")
    parser.add_argument("--pretrained_model", type=str, 
                        default="facebook/hubert-large-ls960-ft", 
                        help="Pretrained HuggingFace model")
    parser.add_argument("--freeze_feature_extractor", action="store_true", 
                        help="Freeze the feature extractor layers")
    parser.add_argument("--feature_dim", type=int, default=128, 
                        help="Dimension of features from each modality")
    
    # Evaluation parameters
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate visualizations")
    
    return parser.parse_args()

def get_dataset_paths(args):
    """
    Get dataset paths based on arguments
    """
    train_wav_dirs = [args.train_wav]
    train_json_dirs = [args.train_json]
    test_wav_dirs = [args.test_wav]
    test_json_dirs = [args.test_json]
    
    # Include 2023 data if specified
    if args.include_2023:
        
        test_wav_dirs.append(args.test_wav + "_2023")
        test_json_dirs.append(args.test_json + "_2023")
    
    return train_wav_dirs, train_json_dirs, test_wav_dirs, test_json_dirs

def prepare_datasets(args, feature_extractor):
    """
    Prepare datasets for training and validation
    
    Args:
        args: Command line arguments
        feature_extractor: HuBERT feature extractor
    
    Returns:
        train_dataset, val_dataset
    """
    # Get dataset paths
    train_wav_dirs, train_json_dirs, test_wav_dirs, test_json_dirs = get_dataset_paths(args)
    
    # Task type - always use event level for respiratory sounds
    task_type = "event"
    
    # Load training data
    print("Creating training dataset...")
    train_dataset = SPRSoundDataset(
        audio_dirs=train_wav_dirs,
        json_dirs=train_json_dirs,
        feature_extractor=feature_extractor,
        task=task_type,
        transform=True,
        max_length=Config.MAX_LENGTH
    )
    
    # Check for test data
    print("Creating validation dataset...")
    val_datasets = []
    
    # Try to load test data from specified directories
    for wav_path, json_path in zip(test_wav_dirs, test_json_dirs):
        if os.path.exists(wav_path) and os.path.exists(json_path):
            try:
                test_data = SPRSoundDataset(
                    audio_dirs=wav_path,
                    json_dirs=json_path,
                    feature_extractor=feature_extractor,
                    task=task_type,
                    transform=False,
                    max_length=Config.MAX_LENGTH
                )
                if len(test_data) > 0:
                    print(f"Loaded {len(test_data)} samples from {wav_path}")
                    val_datasets.append(test_data)
            except Exception as e:
                print(f"Error loading test data from {wav_path}: {e}")
    
    # Combine test datasets if available
    if val_datasets:
        val_dataset = ConcatDataset(val_datasets)
    else:
        # If no test data, split training data
        print("No test data found. Using a portion of training data for validation.")
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        
        train_dataset = Subset(train_dataset, indices[:train_size])
        val_dataset = Subset(train_dataset.dataset, indices[train_size:])
        # Disable augmentation for validation
        val_dataset.dataset.transform = False
    
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset

def create_data_loaders(train_dataset, val_dataset, batch_size):
    """
    Create data loaders with balanced sampling for the training set
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader
    """
    # Create balanced sampling weights
    sample_weights = None
    if hasattr(train_dataset, '_create_balanced_sampling_weights'):
        sample_weights = train_dataset._create_balanced_sampling_weights()
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, '_create_balanced_sampling_weights'):
        all_weights = train_dataset.dataset._create_balanced_sampling_weights()
        # If train_dataset is a Subset, use indices
        if hasattr(train_dataset, 'indices'):
            sample_weights = [all_weights[i] for i in train_dataset.indices]
        else:
            sample_weights = all_weights
    
    # Create sampler and data loader
    if sample_weights:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    # Create validation loader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

def evaluate_model(model, test_loader, device, output_dir):
    """
    Evaluate the hierarchical model on the test set
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        output_dir: Directory to save results
    
    Returns:
        Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    binary_preds = []
    binary_labels = []
    modality_attentions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_values, return_attention=True)
            
            # Get predictions
            preds = model.predict(outputs)
            
            # Convert to binary predictions
            binary_pred = torch.where(preds > 0, 1, 0)
            binary_label = torch.where(labels > 0, 1, 0)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            binary_preds.extend(binary_pred.cpu().numpy())
            binary_labels.extend(binary_label.cpu().numpy())
            
            # Store modality attention weights
            if 'modality_attention' in outputs:
                modality_attentions.extend(outputs['modality_attention'].cpu().numpy())
    
    # Calculate overall metrics
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Calculate binary metrics
    binary_accuracy = np.mean(np.array(binary_labels) == np.array(binary_preds))
    binary_f1 = f1_score(binary_labels, binary_preds, average='binary', zero_division=0)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save confusion matrix visualization
    class_labels = ["Normal", "Rhonchi", "Wheeze", "Stridor", "Coarse Crackle", "Fine Crackle", "Wheeze & Crackle"]
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Final Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    plt.tight_layout()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix_final.png"))
    
    # Binary confusion matrix
    binary_cm = confusion_matrix(binary_labels, binary_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(binary_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Binary Confusion Matrix (Normal vs Adventitious)")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal", "Adventitious"])
    plt.yticks([0, 1], ["Normal", "Adventitious"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "binary_confusion_matrix.png"))
    
    # If we have modality attention, visualize it
    if modality_attentions:
        modality_attention_avg = np.mean(np.array(modality_attentions), axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.bar(['HuBERT', 'Spectrogram', 'MFCC'], modality_attention_avg)
        plt.title("Average Modality Attention Weights")
        plt.ylabel("Attention Weight")
        plt.ylim(0, 1)
        for i, value in enumerate(modality_attention_avg):
            plt.text(i, value + 0.01, f"{value:.3f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "modality_attention.png"))
    
    # Compile metrics
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(f1),
        'binary_accuracy': float(binary_accuracy),
        'binary_f1': float(binary_f1),
        'confusion_matrix': cm.tolist(),
        'binary_confusion_matrix': binary_cm.tolist()
    }
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print(f"Binary Accuracy (Normal vs Adventitious): {binary_accuracy:.4f}")
    print(f"Binary F1 Score: {binary_f1:.4f}")
    
    return metrics

def main():
    """
    Main function to run training and evaluation
    """
    args = parse_arguments()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load feature extractor (for compatibility with existing data pipeline)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.pretrained_model)
    
    # Prepare output directory
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, "visualizations"), exist_ok=True)
    
    # Save configuration
    config = {
        "pretrained_model": args.pretrained_model,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "freeze_feature_extractor": args.freeze_feature_extractor,
        "feature_dim": args.feature_dim,
        "device": str(device),
        "approach": "hierarchical"
    }
    
    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(args, feature_extractor)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, args.batch_size
    )
    
    # Initialize hierarchical model
    model = HierarchicalRespiratoryClassifier(
        pretrained_model=args.pretrained_model,
        freeze_feature_extractor=args.freeze_feature_extractor,
        feature_dim=args.feature_dim,
        sample_rate=16000  # Standard sample rate
    )
    
    # Print model summary
    print(f"\nHierarchical Respiratory Sound Classifier:")
    print(f"  Pretrained model: {args.pretrained_model}")
    print(f"  Feature dim: {args.feature_dim}")
    print(f"  Training dataset: {len(train_dataset)} samples")
    print(f"  Validation dataset: {len(val_dataset)} samples")
    print(f"  Using hierarchical classification: Normal vs Adventitious → Specific type")
    
    # Train model
    print(f"\nStarting hierarchical training...")
    best_metrics = train_hierarchical_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.model_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-4,
        patience=args.patience
    )
    
    # Re-load best model for evaluation
    best_model_path = os.path.join(args.model_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {best_model_path}")
    
    # Evaluate model
    print("\nEvaluating model on validation set...")
    metrics = evaluate_model(
        model=model,
        test_loader=val_loader,
        device=device,
        output_dir=args.model_dir
    )
    
    # Final report
    print("\nTraining and evaluation complete!")
    print(f"Model saved to {args.model_dir}")
    print(f"Best validation metrics:")
    print(f"  Loss: {best_metrics['loss']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {best_metrics['f1']:.4f}")
    print(f"  Binary Accuracy: {best_metrics['binary_accuracy']:.4f}")
    print(f"Final test metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['macro_f1']:.4f}")
    print(f"  Binary Accuracy: {metrics['binary_accuracy']:.4f}")
    print(f"  Binary F1 Score: {metrics['binary_f1']:.4f}")

if __name__ == "__main__":
    main()