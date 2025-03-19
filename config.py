class Config:
    """
    Minimal configuration for hierarchical respiratory sound classification
    """
    # HuBERT parameters
    SAMPLING_RATE = 16000
    MAX_LENGTH = 8000  # 0.5 seconds of audio at 16kHz
    
    # Event-level classification
    EVENT_CLASSES = {
        'Normal': 0,
        'Rhonchi': 1,
        'Wheeze': 2,
        'Stridor': 3,
        'Coarse Crackle': 4,
        'Fine Crackle': 5,
        'Wheeze & Crackle': 6
    }
    
    # Record-level classification
    RECORD_CLASSES = {
        'Normal': 0,
        'Poor Quality': 1,
        'CAS': 2,
        'DAS': 3,
        'CAS & DAS': 4
    }