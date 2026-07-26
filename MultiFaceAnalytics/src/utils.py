import os
import urllib.request
import cv2
import numpy as np

def detect_device():
    """
    Detects if GPU acceleration is available.
    Returns:
        bool: True if GPU is available (CUDA / DirectML / OpenCL), False otherwise.
        str: Description of the hardware backend.
    """
    gpu_available = False
    backend = "CPU"
    
    # 1. Check ONNX Runtime execution providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            gpu_available = True
            backend = "GPU (ONNX Runtime - CUDA)"
            return gpu_available, backend
        elif "DmlExecutionProvider" in providers:
            gpu_available = True
            backend = "GPU (ONNX Runtime - DirectML)"
            return gpu_available, backend
    except ImportError:
        pass

    # 2. Check OpenCV DNN GPU support
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            gpu_available = True
            backend = f"GPU (OpenCV CUDA - {count} Device(s))"
            return gpu_available, backend
    except AttributeError:
        # OpenCV built without CUDA support
        pass

    return gpu_available, backend

def download_file(url, dest_path):
    """
    Downloads a file from a URL with basic progress reporting.
    Handles network failures gracefully.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} to {dest_path}...")
    try:
        def progress_hook(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = min(100, (read_so_far * 100) // total_size)
                print(f"\rDownloading: {percent}% ({read_so_far // 1024} KB / {total_size // 1024} KB)", end="")
            else:
                print(f"\rDownloading: {read_so_far // 1024} KB", end="")
        
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print("\nDownload finished successfully!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False

def draw_glass_panel(img, x, y, w, h, title, stats, alpha=0.6, border_color=(100, 100, 100)):
    """
    Draws a translucent glassmorphic panel on the image.
    This creates a modern HUD feel.
    """
    # Create the background card overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1)
    # Blend overlay with original
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw a thin borders
    cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 1)
    
    # Draw Title
    cv2.putText(img, title.upper(), (x + 15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 180, 255), 1, cv2.LINE_AA)
    
    # Draw Stats list
    start_y = y + 50
    for key, value in stats.items():
        text_key = f"{key}:"
        text_val = f" {value}"
        cv2.putText(img, text_key, (x + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (200, 200, 200), 1, cv2.LINE_AA)
        # Position the value offset
        key_width = cv2.getTextSize(text_key, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(img, text_val, (x + 15 + key_width, start_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        start_y += 18

def draw_styled_bbox(img, bbox, label_lines, color, thickness=1, corner_len=15):
    """
    Draws a sleek modern bounding box around a face.
    It features light box borders with thicker, bright corner highlights
    and a translucent info card on the side or top.
    """
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    
    # Draw basic box outline (thin)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw double thick corner brackets
    c_color = color
    c_thick = thickness + 2
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), c_color, c_thick)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), c_color, c_thick)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), c_color, c_thick)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), c_color, c_thick)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), c_color, c_thick)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), c_color, c_thick)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), c_color, c_thick)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), c_color, c_thick)

    # Info card: decide to draw on the right, or top if no room
    # We will draw a translucent card next to the box
    card_w = 160
    card_h = 20 + len(label_lines) * 16
    
    card_x = x2 + 5
    card_y = y1
    
    # Boundary check (make sure card fits on screen)
    img_h, img_w = img.shape[:2]
    if card_x + card_w > img_w:
        card_x = x1 - card_w - 5 # Draw on left
        if card_x < 0:
            card_x = max(5, x1) # fallback overlay top inside
            card_y = max(5, y1 - card_h)
            
    if card_y + card_h > img_h:
        card_y = max(5, img_h - card_h - 5)

    # Draw translucent backdrop
    overlay = img.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    
    # Draw border card
    cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)
    
    # Print label lines
    text_y = card_y + 16
    for i, line in enumerate(label_lines):
        # First line (Face ID) is bold/larger
        if i == 0:
            cv2.putText(img, line, (card_x + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, line, (card_x + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.38, (220, 220, 220), 1, cv2.LINE_AA)
        text_y += 15

def get_unique_color(track_id):
    """
    Generates a unique but aesthetically pleasing vibrant color based on face ID.
    Avoids boring/harsh colors. Uses HSV mapping.
    """
    np.random.seed(track_id * 13)
    hue = int(np.random.randint(0, 180))
    # Keep saturation and value high for visibility
    hsv = np.uint8([[[hue, 200, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, bgr))
