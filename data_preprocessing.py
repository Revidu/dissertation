import os
import json
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple, Optional, Union

class SPRSoundDataset(Dataset):
    """
    Custom Dataset for SPRSound Respiratory Sound Classification
    Handles both event-level and record-level annotations
    """
    def __init__(
        self, 
        audio_dirs: Union[str, List[str]], 
        json_dirs: Union[str, List[str]],
        feature_extractor: Wav2Vec2FeatureExtractor,
        task: str = 'event',
        transform: bool = True,
        max_length: int = 8000  # 0.5 second at 16kHz
    ):
        """
        Initialize dataset
        
        Args:
            audio_dirs: Directory or list of directories containing WAV files
            json_dirs: Directory or list of directories containing JSON annotations
            feature_extractor: HuBERT feature extractor
            task: 'event' or 'record' level classification
            transform: Apply data augmentation
            max_length: Maximum length of audio (in samples)
        """
        # Handle single directory or list of directories
        self.audio_dirs = [audio_dirs] if isinstance(audio_dirs, str) else audio_dirs
        self.json_dirs = [json_dirs] if isinstance(json_dirs, str) else json_dirs
        
        self.feature_extractor = feature_extractor
        self.task = task
        self.transform = transform
        self.max_length = max_length
        
        # Event-level classification mappings
        self.event_classes = {
            'Normal': 0,
            'Rhonchi': 1,
            'Wheeze': 2,
            'Stridor': 3,
            'Coarse Crackle': 4,
            'Fine Crackle': 5,
            'Wheeze & Crackle': 6
        }
        
        # Record-level classification mappings
        self.record_classes = {
            'Normal': 0,
            'Poor Quality': 1,
            'CAS': 2,
            'DAS': 3,
            'CAS & DAS': 4
        }
        
        # Prepare dataset
        self.audio_files, self.labels, self.record_labels = self._prepare_dataset()
        print(f"Loaded dataset with {len(self.audio_files)} samples")
        
        # Calculate and print class distribution
        self._print_class_distribution()
    
    def _print_class_distribution(self) -> None:
        """
        Print the class distribution of the dataset
        """
        label_counts = {}
        for label in self.labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        # Convert to class names
        event_class_names = {v: k for k, v in self.event_classes.items()}
        print("Class distribution:")
        for label, count in sorted(label_counts.items()):
            class_name = event_class_names.get(label, f"Unknown ({label})")
            print(f"  {class_name}: {count} samples ({count/len(self.labels)*100:.1f}%)")
    
    def _create_balanced_sampling_weights(self) -> List[float]:
        """
        Create sample weights for balanced sampling
        """
        # Count samples per class
        label_counts = {}
        for i in range(len(self.labels)):
            label = self.labels[i]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        # Calculate weights inversely proportional to class frequency
        max_count = max(label_counts.values())
        weights = []
        for i in range(len(self.labels)):
            label = self.labels[i]
            count = label_counts[label]
            # Higher weight for minority classes
            weight = max_count / count
            # Additional boost for non-normal classes
            if label != 0:  # Assuming Normal is 0
                weight *= 1.5  # Additional weight for minority classes
            weights.append(weight)
        
        return weights
    
    def _prepare_dataset(self) -> Tuple[List[str], List[int], List[int]]:
        """
        Prepare dataset by scanning audio files and extracting labels
        
        Returns:
            Tuple of audio file paths, event labels, and record labels
        """
        audio_files = []
        event_labels = []
        record_labels = []
        
        # Process all directories
        for audio_dir, json_dir in zip(self.audio_dirs, self.json_dirs):
            if not os.path.exists(audio_dir) or not os.path.exists(json_dir):
                print(f"Warning: Directory not found - Audio: {audio_dir}, JSON: {json_dir}")
                continue
                
            # Get all WAV files
            for filename in os.listdir(audio_dir):
                if filename.endswith('.wav'):
                    # Construct full file path
                    file_path = os.path.join(audio_dir, filename)
                    
                    # Look for corresponding JSON annotation
                    json_filename = filename.replace('.wav', '.json')
                    json_path = os.path.join(json_dir, json_filename)
                    
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                annotation = json.load(f)
                            
                            # Get record-level label
                            record_label = annotation.get('recording_annotation', 'Normal')
                            record_label_idx = self.record_classes.get(record_label, 0)
                            
                            # Get event-level label (use first event)
                            event_annotations = annotation.get('event_annotation', [])
                            if event_annotations:
                                event_type = event_annotations[0].get('type', 'Normal')
                                event_label_idx = self.event_classes.get(event_type, 0)
                            else:
                                event_label_idx = 0  # Default to Normal if no events
                            
                            audio_files.append(file_path)
                            event_labels.append(event_label_idx)
                            record_labels.append(record_label_idx)
                        except Exception as e:
                            print(f"Error processing {json_path}: {e}")
        
        return audio_files, event_labels, record_labels
    
    def _load_audio(self, file_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Preprocessed audio tensor
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Resample to 16kHz for HuBERT
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Ensure proper length for HuBERT
            if waveform.shape[1] > self.max_length:
                # Take the middle portion of the audio
                start = (waveform.shape[1] - self.max_length) // 2
                waveform = waveform[:, start:start + self.max_length]
            
            # Pad if too short
            if waveform.shape[1] < self.max_length:
                padding = torch.zeros(1, self.max_length - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)
            
            return waveform.squeeze()
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            # Return a zero tensor as fallback with correct length for HuBERT
            return torch.zeros(self.max_length)
    
    def _augment_audio(self, waveform: torch.Tensor, label: int = 0) -> torch.Tensor:
        """
        Apply augmentation for better training
        
        Args:
            waveform: Input audio waveform
            label: Class label to apply appropriate augmentation
            
        Returns:
            Augmented audio waveform
        """
        # Only apply augmentation randomly
        if np.random.random() < 0.5:
            # Choose an augmentation
            aug_type = np.random.choice(['noise', 'time_shift', 'gain'])
            
            if aug_type == 'noise':
                # More noise for minority classes, less for normal
                noise_level = 0.01 if label != 0 else 0.002
                noise = torch.randn_like(waveform) * noise_level
                waveform = waveform + noise
            
            elif aug_type == 'time_shift':
                # Shift audio slightly
                shift = int(np.random.uniform(-0.1, 0.1) * len(waveform))
                if shift > 0:
                    waveform = torch.cat([waveform[-shift:], waveform[:-shift]])
                elif shift < 0:
                    waveform = torch.cat([waveform[-shift:], waveform[:-shift]])
            
            elif aug_type == 'gain':
                # Random gain (very fast operation)
                gain = np.random.uniform(0.8, 1.2)
                waveform = waveform * gain
        
        # Normalize waveform
        if torch.abs(waveform).max() > 0:
            waveform = waveform / torch.abs(waveform).max()
        
        return waveform
    
    def __len__(self) -> int:
        """
        Get dataset length
        
        Returns:
            Number of audio samples
        """
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset
        
        Args:
            idx: Index of the item
        
        Returns:
            Dictionary containing processed audio and labels
        """
        # Load audio
        waveform = self._load_audio(self.audio_files[idx])
        
        # Apply augmentation if enabled
        if self.transform:
            try:
                # Pass the label to apply class-specific augmentation
                waveform = self._augment_audio(waveform, self.labels[idx])
            except Exception as e:
                print(f"Augmentation failed: {e}. Using original waveform.")
        
        # Extract features using HuBERT feature extractor
        try:
            # Detach the tensor before converting to numpy
            if hasattr(waveform, 'requires_grad') and waveform.requires_grad:
                waveform_np = waveform.detach().numpy()
            else:
                waveform_np = waveform.numpy()
                
            inputs = self.feature_extractor(
                waveform_np, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # Extract the tensor from batch dimension
            input_values = inputs.input_values.squeeze(0)
        except Exception as e:
            print(f"Feature extraction failed: {e}. Using fallback.")
            # Fallback to empty features
            input_values = torch.zeros(self.max_length)
        
        # Return based on task
        if self.task == 'event':
            return {
                'input_values': input_values,
                'label': self.labels[idx],
                'record_label': self.record_labels[idx]  # Include record label for multi-task learning
            }
        else:  # record-level task
            return {
                'input_values': input_values,
                'label': self.record_labels[idx],
                'event_label': self.labels[idx]  # Include event label for multi-task learning
            }