import os
import argparse
import numpy as np

def generate_placeholder_onnx(models_dir):
    """
    Generates dummy/placeholder ONNX models using PyTorch.
    This ensures that ONNX runtime and OpenCV DNN have valid, loadable model files.
    """
    print("Attempting to generate placeholder ONNX models using PyTorch...")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("\n[WARNING] PyTorch is not installed. To generate placeholder ONNX models, please run:")
        print("  pip install torch torchvision")
        print("The application will fallback to landmark-based heuristic analytics in the meantime.\n")
        return False

    os.makedirs(models_dir, exist_ok=True)

    # 1. Define and export Emotion Model (Input: 1x1x48x48, Output: 1x7)
    class EmotionNet(nn.Module):
        def __init__(self):
            super(EmotionNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 6 * 6, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 7) # 7 emotions
            )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    emotion_path = os.path.join(models_dir, "emotion_model.onnx")
    print(f"Creating placeholder Emotion model at: {emotion_path}")
    emotion_model = EmotionNet()
    emotion_model.eval()
    dummy_emotion_input = torch.randn(1, 1, 48, 48, requires_grad=False)
    torch.onnx.export(
        emotion_model,
        dummy_emotion_input,
        emotion_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Emotion model exported successfully!")

    # 2. Define and export Age/Gender Model (Input: 1x3x224x224, Outputs: age (1x1) & gender (1x2))
    class AgeGenderNet(nn.Module):
        def __init__(self):
            super(AgeGenderNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d((7, 7))
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU()
            )
            self.age_head = nn.Sequential(
                nn.Linear(128, 1) # Age regression value
            )
            self.gender_head = nn.Sequential(
                nn.Linear(128, 2) # 2 classes: Female, Male
            )
        def forward(self, x):
            features = self.fc(self.features(x))
            age = self.age_head(features)
            gender = self.gender_head(features)
            return age, gender

    age_gender_path = os.path.join(models_dir, "age_gender_model.onnx")
    print(f"Creating placeholder Age/Gender model at: {age_gender_path}")
    age_gender_model = AgeGenderNet()
    age_gender_model.eval()
    dummy_age_gender_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    torch.onnx.export(
        age_gender_model,
        dummy_age_gender_input,
        age_gender_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['age_output', 'gender_output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'age_output': {0: 'batch_size'},
            'gender_output': {0: 'batch_size'}
        }
    )
    print("Age/Gender model exported successfully!")

    # 3. Define and export Hand Sign Model (Input: 1x63, Output: 1x5)
    class HandSignNet(nn.Module):
        def __init__(self, num_classes=5):
            super(HandSignNet, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(63, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        def forward(self, x):
            return self.fc(x)

    hand_sign_path = os.path.join(models_dir, "hand_sign_model.onnx")
    print(f"Creating placeholder Hand Sign model at: {hand_sign_path}")
    hand_sign_model = HandSignNet()
    hand_sign_model.eval()
    dummy_hand_input = torch.randn(1, 63, requires_grad=False)
    torch.onnx.export(
        hand_sign_model,
        dummy_hand_input,
        hand_sign_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Hand Sign model exported successfully!")
    return True

def generate_synthetic_gestures(num_samples=250):
    """
    Generates synthetic hand landmarks (21 points, 3D) representing 5 signs:
    0: Fist, 1: Open Palm, 2: Victory, 3: Thumbs Up, 4: Thumbs Down
    """
    print(f"Generating {num_samples * 5} synthetic gesture landmark samples...")
    X = []
    y = []
    
    for label in range(5):
        for _ in range(num_samples):
            landmarks = np.zeros((21, 3), dtype=np.float32)
            
            # Base MCPs
            landmarks[1] = [0.15, -0.05, 0.0]
            landmarks[2] = [0.25, -0.1, 0.0]
            landmarks[5] = [0.15, -0.25, 0.0]
            landmarks[9] = [0.05, -0.27, 0.0]
            landmarks[13] = [-0.05, -0.25, 0.0]
            landmarks[17] = [-0.15, -0.22, 0.0]
            
            extended = [False] * 5 # Thumb, Index, Middle, Ring, Pinky
            if label == 0:    # Fist
                extended = [False, False, False, False, False]
            elif label == 1:  # Open Palm
                extended = [True, True, True, True, True]
            elif label == 2:  # Victory
                extended = [False, True, True, False, False]
            elif label == 3:  # Thumbs Up
                extended = [True, False, False, False, False]
            elif label == 4:  # Thumbs Down
                extended = [True, False, False, False, False]
            
            # Thumb IP/TIP
            if label == 3: # Thumbs Up (Thumb points up)
                landmarks[3] = [0.25, -0.2, 0.0]
                landmarks[4] = [0.25, -0.3, 0.0]
            elif label == 4: # Thumbs Down (Thumb points down)
                landmarks[3] = [0.25, 0.0, 0.0]
                landmarks[4] = [0.25, 0.1, 0.0]
            else:
                if extended[0]: # Thumb extended side
                    landmarks[3] = [0.35, -0.12, 0.0]
                    landmarks[4] = [0.45, -0.14, 0.0]
                else: # Thumb folded
                    landmarks[3] = [0.15, -0.12, 0.0]
                    landmarks[4] = [0.08, -0.12, 0.0]
            
            # Four fingers
            for f_idx, start_lm in enumerate([5, 9, 13, 17]):
                mcp = landmarks[start_lm]
                if extended[f_idx + 1]:
                    # Extended: extend upwards (along negative Y)
                    landmarks[start_lm + 1] = mcp + [0.0, -0.10, 0.0]
                    landmarks[start_lm + 2] = mcp + [0.0, -0.18, 0.0]
                    landmarks[start_lm + 3] = mcp + [0.0, -0.25, 0.0]
                else:
                    # Folded: curve back towards palm (positive Y / inside)
                    landmarks[start_lm + 1] = mcp + [0.0, 0.05, 0.0]
                    landmarks[start_lm + 2] = mcp + [0.0, 0.10, 0.0]
                    landmarks[start_lm + 3] = mcp + [0.0, 0.08, 0.0]
            
            # Apply random scaling
            scale = np.random.uniform(0.85, 1.15)
            landmarks *= scale
            
            # Apply random rotation around Z axis
            angle = np.random.uniform(-0.25, 0.25)
            if label == 4: # Thumbs down is rotated 180 degrees relative to Thumbs up
                angle += np.pi
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
            landmarks = np.dot(landmarks, R.T)
            
            # Add noise
            noise = np.random.normal(0, 0.012, size=landmarks.shape).astype(np.float32)
            landmarks += noise
            
            # Center around wrist
            landmarks_rel = landmarks - landmarks[0]
            
            # Scale-normalize by max distance
            max_dist = np.max(np.linalg.norm(landmarks_rel, axis=1))
            if max_dist > 0:
                landmarks_norm = landmarks_rel / max_dist
            else:
                landmarks_norm = landmarks_rel
                
            X.append(landmarks_norm.flatten())
            y.append(label)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def train_hand_sign_model(models_dir, epochs=15):
    """
    Trains a neural network classifier to detect hand signs from landmark coordinate structures.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("[ERROR] PyTorch is required to run the hand sign model training loop.")
        return False

    print(f"Training Hand Sign model for {epochs} epochs...")
    
    # Generate dataset
    X_train, y_train = generate_synthetic_gestures(num_samples=400)
    
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    class HandSignNet(nn.Module):
        def __init__(self, num_classes=5):
            super(HandSignNet, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(63, 64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        def forward(self, x):
            return self.fc(x)

    model = HandSignNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | Accuracy: {100.0 * correct/total:.2f}%")
        
    # Export to ONNX
    os.makedirs(models_dir, exist_ok=True)
    hand_sign_path = os.path.join(models_dir, "hand_sign_model.onnx")
    model.eval()
    dummy_input = torch.randn(1, 63)
    torch.onnx.export(
        model, 
        dummy_input, 
        hand_sign_path, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully trained and exported Hand Sign model to: {hand_sign_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiFaceAnalytics Training & Model Generation Utility")
    parser.add_argument("--mode", type=str, default="placeholder", choices=["placeholder", "train_gesture"],
                        help="Mode: 'placeholder' to generate all fast untargeted models, 'train_gesture' to run actual hand signs training.")
    parser.add_argument("--models-dir", type=str, default="./models", help="Directory where models should be saved.")
    args = parser.parse_args()

    # Convert relative path to absolute depending on execution
    abs_models_dir = os.path.abspath(args.models_dir)
    print(f"Target directory for models: {abs_models_dir}")

    if args.mode == "placeholder":
        generate_placeholder_onnx(abs_models_dir)
    elif args.mode == "train_gesture":
        train_hand_sign_model(abs_models_dir)
