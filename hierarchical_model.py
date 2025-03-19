import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import HubertModel
from typing import Dict, Tuple, Optional, Union, List

class HierarchicalRespiratoryClassifier(nn.Module):
    """
    Hierarchical respiratory sound classifier that:
    1. First classifies sounds as Normal vs Adventitious
    2. Then classifies Adventitious sounds into specific types
    
    Uses the same multi-modal architecture as before but with a hierarchical output structure.
    """
    def __init__(
        self,
        pretrained_model: str = "facebook/hubert-large-ls960-ft",
        freeze_feature_extractor: bool = True,
        feature_dim: int = 128,
        sample_rate: int = 16000
    ):
        """
        Initialize the hierarchical model
        
        Args:
            pretrained_model: HuggingFace model ID for HuBERT
            freeze_feature_extractor: Whether to freeze HuBERT feature extractor
            feature_dim: Dimension of features from each modality
            sample_rate: Audio sample rate
        """
        super().__init__()
        
        # 1. Audio processing with HuBERT
        self.hubert = HubertModel.from_pretrained(pretrained_model)
        self.hidden_size = self.hubert.config.hidden_size  # Usually 768 for HuBERT base, 1024 for large
        
        # Freeze feature extractor if specified
        if freeze_feature_extractor:
            # Freeze all HuBERT parameters
            for param in self.hubert.parameters():
                param.requires_grad = False
            
            # Unfreeze the last few transformer layers for fine-tuning
            unfreeze_layers = 3  # Reduced for less overfitting
            for i in range(12 - unfreeze_layers, 12):  # Assuming 12 layers total
                if hasattr(self.hubert.encoder, f'layer.{i}'):
                    for param in getattr(self.hubert.encoder, f'layer.{i}').parameters():
                        param.requires_grad = True
                elif hasattr(self.hubert.encoder, 'layers') and i < len(self.hubert.encoder.layers):
                    for param in self.hubert.encoder.layers[i].parameters():
                        param.requires_grad = True
        
        # HuBERT projection to feature_dim
        self.hubert_projection = nn.Sequential(
            nn.Linear(self.hidden_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.5)
        )
        
        # 2. Spectrogram processing
        self.spec_extractor = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.Dropout(0.5)  # Increased from 0.3 to 0.5
        )
        
        # 3. MFCC processing
        self.mfcc_extractor = nn.Sequential(
            # First convolutional block
            nn.Conv1d(40, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.Dropout(0.5)  # Increased from 0.3 to 0.5
        )
        
        # 4. Feature fusion with attention
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # 5. Hierarchical classification heads
        # Stage 1: Normal vs Adventitious
        self.binary_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.3 to 0.5
            nn.Linear(128, 2)  # 0: Normal, 1: Adventitious
        )
        
        # Stage 2: Specific adventitious sound type (6 classes, excluding Normal)
        self.adventitious_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),  # Increased from 0.4 to 0.6
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.3 to 0.5
            nn.Linear(128, 6)  # 6 adventitious classes
        )
        
        # Audio transforms for spectrogram and MFCC extraction
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={
                'n_fft': 1024,
                'hop_length': 512,
                'n_mels': 128,
                'f_min': 20,
                'f_max': 8000
            }
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Store class information
        self.stage1_labels = ["Normal", "Adventitious"]
        self.stage2_labels = [
            "Rhonchi", "Wheeze", "Stridor", 
            "Coarse Crackle", "Fine Crackle", "Wheeze & Crackle"
        ]
        
        # Map from original class index to stage1 and stage2 indices
        self.class_mapping = {
            0: (0, None),  # Normal -> (Normal, None)
            1: (1, 0),     # Rhonchi -> (Adventitious, Rhonchi)
            2: (1, 1),     # Wheeze -> (Adventitious, Wheeze)
            3: (1, 2),     # Stridor -> (Adventitious, Stridor)
            4: (1, 3),     # Coarse Crackle -> (Adventitious, Coarse Crackle)
            5: (1, 4),     # Fine Crackle -> (Adventitious, Fine Crackle)
            6: (1, 5)      # Wheeze & Crackle -> (Adventitious, Wheeze & Crackle)
        }
        
        # Inverse mapping from (stage1, stage2) to original class index
        self.inverse_mapping = {
            (0, None): 0,  # (Normal, None) -> Normal
            (1, 0): 1,     # (Adventitious, Rhonchi) -> Rhonchi
            (1, 1): 2,     # (Adventitious, Wheeze) -> Wheeze
            (1, 2): 3,     # (Adventitious, Stridor) -> Stridor
            (1, 3): 4,     # (Adventitious, Coarse Crackle) -> Coarse Crackle
            (1, 4): 5,     # (Adventitious, Fine Crackle) -> Fine Crackle
            (1, 5): 6      # (Adventitious, Wheeze & Crackle) -> Wheeze & Crackle
        }
        
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
    
    def extract_audio_features(self, waveform):
        """
        Extract spectrogram and MFCC features from audio waveform
        
        Args:
            waveform: Audio waveform tensor [batch_size, time]
        
        Returns:
            mel_specs: Mel spectrogram features
            mfccs: MFCC features
        """
        batch_size = waveform.size(0)
        device = waveform.device
        
        # Ensure waveform is proper shape for processing
        if waveform.dim() == 2 and waveform.size(0) == batch_size:
            # Already in [batch_size, time] format
            pass
        elif waveform.dim() == 3 and waveform.size(0) == batch_size and waveform.size(1) == 1:
            # In [batch_size, channels, time] format with 1 channel
            waveform = waveform.squeeze(1)
        else:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")
        
        # Compute spectrograms and MFCCs in batch
        try:
            # Add channel dimension for spectrograms: [batch_size, 1, time]
            waveform_for_spec = waveform.unsqueeze(1)
            
            # Extract mel spectrograms
            mel_specs = self.mel_spec_transform(waveform_for_spec)  # [batch_size, 1, n_mels, time]
            mel_specs = self.amplitude_to_db(mel_specs)  # Convert to dB scale
            
            # Extract MFCCs
            mfccs = self.mfcc_transform(waveform_for_spec)  # [batch_size, n_mfcc, time]
            
            return mel_specs, mfccs
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            # Return empty tensors as fallback
            return (
                torch.zeros(batch_size, 1, 128, 50).to(device),  # Empty spectrograms
                torch.zeros(batch_size, 40, 50).to(device)       # Empty MFCCs
            )
    
    def forward(
        self, 
        input_values: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the hierarchical model
        
        Args:
            input_values: Input audio waveform
            attention_mask: Attention mask for HuBERT
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary of outputs
        """
        batch_size = input_values.size(0)
        device = input_values.device
        
        # 1. Extract HuBERT features
        hubert_outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=return_attention
        )
        
        hubert_hidden_states = hubert_outputs.last_hidden_state
        hubert_attention = hubert_outputs.attentions[-1] if return_attention and hubert_outputs.attentions else None
        
        # Global average pooling over sequence length
        hubert_pooled = torch.mean(hubert_hidden_states, dim=1)
        hubert_features = self.hubert_projection(hubert_pooled)
        
        # 2. Extract and process spectrogram and MFCC features
        mel_specs, mfccs = self.extract_audio_features(input_values)
        
        # Ensure MFCC has the right shape
        if mfccs.dim() == 4:  # If shape is [batch_size, channels, features, time]
            mfccs = mfccs.squeeze(1)  # Remove channel dimension -> [batch_size, features, time]
        
        # Process spectrograms and MFCCs
        spec_features = self.spec_extractor(mel_specs)
        mfcc_features = self.mfcc_extractor(mfccs)
        
        # 3. Fuse features using attention
        # Concatenate all features
        concat_features = torch.cat([hubert_features, spec_features, mfcc_features], dim=1)
        
        # Calculate attention weights
        attention_weights = self.attention(concat_features)  # [batch_size, 3]
        
        # Apply attention weights to each modality
        hubert_weighted = hubert_features * attention_weights[:, 0].unsqueeze(1)
        spec_weighted = spec_features * attention_weights[:, 1].unsqueeze(1)
        mfcc_weighted = mfcc_features * attention_weights[:, 2].unsqueeze(1)
        
        # Sum weighted features
        fused_features = hubert_weighted + spec_weighted + mfcc_weighted
        
        # 4. Apply hierarchical classification
        # Stage 1: Normal vs Adventitious
        binary_logits = self.binary_classifier(fused_features)
        
        # Stage 2: Specific adventitious sound type
        adventitious_logits = self.adventitious_classifier(fused_features)
        
        # Return outputs
        outputs = {
            'binary_logits': binary_logits,  # Normal vs Adventitious
            'adventitious_logits': adventitious_logits,  # Specific adventitious type
            'pooled_features': fused_features,
            'modality_attention': attention_weights,
        }
        
        if return_attention:
            outputs['hubert_attention'] = hubert_attention
        
        return outputs
    
    def predict(self, outputs, threshold=0.5):
        """
        Make hierarchical predictions
        
        Args:
            outputs: Model outputs
            threshold: Threshold for binary classification confidence
        
        Returns:
            Original class predictions (0-6)
        """
        binary_probs = F.softmax(outputs['binary_logits'], dim=1)
        adventitious_probs = F.softmax(outputs['adventitious_logits'], dim=1)
        
        # Get binary predictions (0: Normal, 1: Adventitious)
        binary_preds = torch.argmax(binary_probs, dim=1)
        
        # Get adventitious type predictions
        adventitious_preds = torch.argmax(adventitious_probs, dim=1)
        
        # Map back to original class indices
        final_preds = []
        for b, a in zip(binary_preds.cpu().numpy(), adventitious_preds.cpu().numpy()):
            if b == 0:  # Normal
                final_preds.append(0)  # Original class 0 (Normal)
            else:  # Adventitious
                # Map adventitious prediction (0-5) to original class (1-6)
                final_preds.append(a + 1)
        
        return torch.tensor(final_preds, device=binary_preds.device)
    
    def loss_function(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 8.0,  # Increased from 0.75 to 8.0 for better class balancing
        binary_weight: float = 0.5,  # Equal weight now
        adventitious_weight: float = 0.5  # Equal weight now
    ) -> torch.Tensor:
        """
        Hierarchical focal loss function
        
        Args:
            outputs: Model outputs
            labels: Original class labels (0-6)
            gamma: Focal loss gamma parameter
            alpha: Focal loss alpha parameter for balancing classes
            binary_weight: Weight for binary classification loss
            adventitious_weight: Weight for adventitious classification loss
        
        Returns:
            Combined loss
        """
        try:
            # Map original labels to hierarchical labels
            binary_labels = torch.zeros_like(labels)
            adventitious_labels = torch.zeros_like(labels)
            adventitious_mask = torch.zeros_like(labels, dtype=torch.bool)
            
            for i, label in enumerate(labels):
                stage1, stage2 = self.class_mapping[label.item()]
                binary_labels[i] = stage1
                if stage1 == 1:  # Adventitious
                    adventitious_labels[i] = stage2
                    adventitious_mask[i] = True
            
            # Binary (Normal vs Adventitious) loss
            binary_logits = outputs['binary_logits']
            binary_ce = F.cross_entropy(binary_logits, binary_labels, reduction='none')
            pt_binary = torch.exp(-binary_ce)
            binary_focal_loss = (1 - pt_binary) ** gamma * binary_ce
            
            # Apply class weights for binary loss
            binary_weights = torch.ones_like(binary_labels, dtype=torch.float)
            binary_weights[binary_labels == 0] = 1.0  # Normal (majority class)
            binary_weights[binary_labels == 1] = alpha * 6.0  # Adventitious (minority class)
            binary_loss = (binary_focal_loss * binary_weights).mean()
            
            # Adventitious type loss (only for adventitious samples)
            adventitious_loss = 0.0
            if adventitious_mask.sum() > 0:
                adventitious_logits = outputs['adventitious_logits']
                adventitious_logits_masked = adventitious_logits[adventitious_mask]
                adventitious_labels_masked = adventitious_labels[adventitious_mask]
                
                # Calculate class weights based on original dataset distribution
                # Approximate counts from your dataset
                class_counts = {
                    0: 5,    # Rhonchi
                    1: 111,  # Wheeze
                    2: 3,    # Stridor
                    3: 13,   # Coarse Crackle
                    4: 229,  # Fine Crackle
                    5: 10    # Wheeze & Crackle (estimated)
                }
                
                # Calculate weights inversely proportional to frequency
                total = sum(class_counts.values())
                class_weights = torch.zeros(6, device=adventitious_logits.device)
                for cls, count in class_counts.items():
                    class_weights[cls] = total / (count * 6)  # Normalized by number of classes
                
                # Apply weights to loss
                adventitious_ce = F.cross_entropy(
                    adventitious_logits_masked, 
                    adventitious_labels_masked,
                    weight=class_weights,
                    reduction='none'
                )
                
                # Apply focal loss
                pt_adventitious = torch.exp(-adventitious_ce)
                adventitious_focal_loss = (1 - pt_adventitious) ** gamma * adventitious_ce
                adventitious_loss = adventitious_focal_loss.mean()
            
            # Combine losses
            total_loss = binary_weight * binary_loss + adventitious_weight * adventitious_loss
            
            return total_loss
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            # Return a zero tensor with requires_grad=True instead of None
            return torch.tensor(0.0, device=outputs['binary_logits'].device, requires_grad=True)


# Helper function to train the hierarchical model
def train_hierarchical_model(
    model,
    train_loader,
    val_loader,
    device,
    output_dir,
    num_epochs=15,
    learning_rate=1e-5,  # Reduced from 2e-5 to 1e-5
    weight_decay=1e-4,   # Increased from 1e-5 to 1e-4
    patience=4
):
    """
    Train the hierarchical model with early stopping
    
    Args:
        model: HierarchicalRespiratoryClassifier model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        output_dir: Directory to save model
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        patience: Early stopping patience
    
    Returns:
        Dictionary of best validation metrics
    """
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer with parameter groups
    hubert_params = []
    cnn_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'hubert' in name:
                hubert_params.append(param)
            elif 'spec_extractor' in name or 'mfcc_extractor' in name:
                cnn_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': hubert_params, 'lr': learning_rate * 0.1},  # Lower LR for fine-tuning
        {'params': cnn_params, 'lr': learning_rate},
        {'params': other_params, 'lr': learning_rate}
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_val_metrics = None
    patience_counter = 0
    
    print(f"Starting hierarchical training for {num_epochs} epochs with early stopping (patience={patience})...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move data to device
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_values)
            
            # Compute loss
            loss = model.loss_function(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Get predictions
            preds = model.predict(outputs)
            
            # Store for metrics calculation
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        # Also calculate stage 1 (binary) accuracy
        binary_train_labels = [1 if label > 0 else 0 for label in train_labels]
        binary_train_preds = [1 if pred > 0 else 0 for pred in train_preds]
        binary_train_acc = accuracy_score(binary_train_labels, binary_train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Move data to device
                input_values = batch['input_values'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_values)
                
                # Compute loss
                loss = model.loss_function(outputs, labels)
                
                # Update metrics
                val_loss += loss.item()
                
                # Get predictions
                preds = model.predict(outputs)
                
                # Store for metrics calculation
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        # Also calculate stage 1 (binary) accuracy
        binary_val_labels = [1 if label > 0 else 0 for label in val_labels]
        binary_val_preds = [1 if pred > 0 else 0 for pred in val_preds]
        binary_val_acc = accuracy_score(binary_val_labels, binary_val_preds)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"  Binary Train Acc: {binary_train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"  Binary Val Acc: {binary_val_acc:.4f}")
        
        # Create confusion matrix every few epochs
        if (epoch + 1) % 3 == 0 or epoch == 0:
            cm = confusion_matrix(val_labels, val_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues",
                xticklabels=["Normal"] + model.stage2_labels,
                yticklabels=["Normal"] + model.stage2_labels
            )
            plt.title(f"Epoch {epoch+1} - Validation Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualizations", f"confusion_matrix_epoch_{epoch+1}.png"))
            plt.close()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = {
                'loss': val_loss,
                'accuracy': val_acc,
                'f1': val_f1,
                'binary_accuracy': binary_val_acc,
                'epoch': epoch + 1
            }
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_metrics': best_val_metrics
            }, os.path.join(output_dir, "best_model.pt"))
            
            print(f"New best model saved!")
            
            # Reset patience counter
            patience_counter = 0
        else:
            # Increment patience counter
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best val loss: {best_val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save checkpoint only every 5 epochs to save space
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Print best metrics
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_metrics['loss']:.4f}")
    print(f"Best validation accuracy: {best_val_metrics['accuracy']:.4f}")
    print(f"Best validation F1 score: {best_val_metrics['f1']:.4f}")
    print(f"Best validation binary accuracy: {best_val_metrics['binary_accuracy']:.4f}")
    print(f"Best epoch: {best_val_metrics['epoch']}")
    
    return best_val_metrics