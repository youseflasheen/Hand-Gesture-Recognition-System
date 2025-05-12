import cv2
import os
import numpy as np
import time

def create_directories(base_path, gestures):
    """Create directories for each gesture if they don't exist."""
    for gesture in gestures:
        path = os.path.join(base_path, gesture)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

def collect_data(gesture_name, save_path, num_images=5000, delay=0.1):
    """Collect images for a specific gesture."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"\nCollecting images for gesture: {gesture_name}")
    print("Press 'q' to stop collecting images for this gesture")
    print("Press 'c' to capture an image")
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display count on frame
        cv2.putText(frame, f"Count: {count}/{num_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save the image
            image_path = os.path.join(save_path, f"{gesture_name}_{count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved image {count + 1}/{num_images}")
            count += 1
            time.sleep(delay)  # Add delay between captures

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCollected {count} images for {gesture_name}")

def main():
    # Define gestures
    gestures = ['hello', 'yes', 'no', 'i love you', 'thank you']
    
    # Set base path for data collection
    base_path = r"E:\new yousef\hand-gesture-recognition\data\raw"
    
    # Create directories
    create_directories(base_path, gestures)
    
    # Collect data for each gesture
    for gesture in gestures:
        gesture_path = os.path.join(base_path, gesture)
        collect_data(gesture, gesture_path)
        
        # Ask if user wants to continue
        response = input(f"\nDo you want to continue with the next gesture? (y/n): ")
        if response.lower() != 'y':
            break
    
    print("\nData collection completed!")

if __name__ == "__main__":
    main()