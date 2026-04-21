#!/usr/bin/env python3
"""
Script to capture training photos for face recognition.
Captures multiple photos of each person from webcam and saves them in training_data/ folder.
"""

import cv2
import os
import sys

def capture_training_faces():
    """Capture training photos for each person"""
    
    # Create training_data directory if it doesn't exist
    if not os.path.exists("training_data"):
        os.makedirs("training_data")
    
    print("=" * 60)
    print("Face Recognition - Training Photo Capture")
    print("=" * 60)
    print()
    print("This script will help you capture training photos for each person")
    print("you want the system to recognize.")
    print()
    
    # Get list of people to train
    people = []
    print("Enter the names of people you want to train for.")
    print("Press Enter with empty name when done.")
    print()
    
    while True:
        name = input("Enter person name (or press Enter to finish): ").strip()
        if not name:
            break
        if name in people:
            print(f"  ⚠️  '{name}' already added. Skipping.")
            continue
        people.append(name)
        print(f"  ✓ Added '{name}'")
    
    if not people:
        print("\n❌ No people added. Exiting.")
        return
    
    print(f"\n✓ Will capture photos for {len(people)} people: {', '.join(people)}")
    print()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Capture photos for each person
    for person_name in people:
        print("=" * 60)
        print(f"Capturing photos for: {person_name}")
        print("=" * 60)
        
        # Create person directory
        person_dir = os.path.join("training_data", person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        input(f"\nPress Enter when {person_name} is ready in front of camera...")
        
        photos_to_capture = 20
        countdown = 3
        
        print(f"\nWill capture {photos_to_capture} photos in {countdown} seconds...")
        print("Move your head slightly between captures for better training!")
        
        # Countdown
        for i in range(countdown, 0, -1):
            print(f"{i}...")
            cv2.waitKey(1000)
        
        print("\n📸 Capturing photos...")
        
        captured = 0
        while captured < photos_to_capture:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Show the frame
            display_frame = frame.copy()
            info_text = f"{person_name} - Photo {captured + 1}/{photos_to_capture}"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Capturing Training Photos', display_frame)
            
            # Save the photo
            filename = f"{captured + 1}.jpg"
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, frame)
            
            captured += 1
            print(f"  ✓ Captured photo {captured}/{photos_to_capture}")
            
            # Small delay between captures
            cv2.waitKey(200)
        
        print(f"\n✓ Captured {captured} photos for {person_name}")
        print(f"  Saved in: {person_dir}/")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✓ Photo capture complete!")
    print("=" * 60)
    print(f"\nCaptured photos for {len(people)} people:")
    for person in people:
        person_dir = os.path.join("training_data", person)
        count = len([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
        print(f"  • {person}: {count} photos")
    
    print("\nNext step: Run 'python3 train_model.py' to train the model")

if __name__ == "__main__":
    try:
        capture_training_faces()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
