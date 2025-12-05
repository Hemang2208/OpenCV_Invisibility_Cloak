# Required libraries for the invisibility cloak app
import streamlit as st  # Web app framework
import numpy as np      # Array operations
import cv2              # Computer vision operations
import av               # Video frame handling
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration  # WebRTC streaming
import time             # Time operations


# Configure the Streamlit page
st.set_page_config(page_title="Invisibility Cloak", layout="centered")
st.title("🧙 Invisibility Cloak - Harry Potter Edition")

# WebRTC configuration for video streaming
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class CloakProcessor(VideoProcessorBase):
    def __init__(self):
        # Background storage and capture settings
        self.background = None
        self.frames_captured = 0
        self.background_frames_needed = 80  # Number of frames to capture for stable background
        self.background_capturing = True
        self.capture_start_time = time.time()

        # Blue color detection range in HSV color space
        self.lower_blue = np.array([90, 50, 50])   # Lower HSV threshold for blue
        self.upper_blue = np.array([130, 255, 255]) # Upper HSV threshold for blue
        
        # Running average accumulator for stable background
        self.background_accumulator = None
        
    def capture_background(self, frame):
        """Use running average for more efficient background capture."""
        try:
            # Initialize accumulator with first frame
            if self.background_accumulator is None:
                self.background_accumulator = frame.astype(np.float32)
            else:
                # Update background using weighted average (95% old, 5% new)
                cv2.accumulateWeighted(frame, self.background_accumulator, 0.05)
            
            self.frames_captured += 1
            
            # Check if enough frames captured for stable background
            if self.frames_captured >= self.background_frames_needed:
                self.background = self.background_accumulator.astype(np.uint8)
                self.background_capturing = False
                print(f"Background captured after {self.frames_captured} frames!")
                return True
                
        except Exception as e:
            print(f"Error in background capture: {e}")
            # Fallback: use current frame as background
            self.background = frame.copy()
            self.background_capturing = False
            return True
            
        return False

    def recv(self, frame):
        """Process each video frame for invisibility effect"""
        try:
            # Convert frame to OpenCV format
            img = frame.to_ndarray(format="bgr24")
            
            # Phase 1: Background capture
            if self.background_capturing:
                self.capture_background(img)
                
                # Show progress to user
                progress = min(self.frames_captured / self.background_frames_needed, 1.0)
                cv2.putText(img, f"Capturing Background: {int(progress*100)}%", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, "Stand Still!", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Phase 2: Apply invisibility effect
            if self.background is None:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Convert current frame to HSV for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Create mask to identify blue colored objects
            mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
            
            # Clean noise from mask using morphological operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)  # Remove small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2) # Fill small gaps
            
            # Smooth mask edges for better blending
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Prepare mask for 3-channel blending
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0    # Normalize to 0-1
            mask_inv_3d = 1.0 - mask_3d                                 # Inverse mask
            
            # Convert images to float for precise blending
            img_float = img.astype(np.float32)
            bg_float = self.background.astype(np.float32)
            
            # Create invisibility effect: show background where blue is detected
            final = (img_float * mask_inv_3d + bg_float * mask_3d).astype(np.uint8)
            
            # Display status to user
            cv2.putText(final, "Invisibility Active!", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(final, format="bgr24")
            
        except Exception as e:
            print(f"Error in frame processing: {e}")
            # Fallback: return original frame if processing fails
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# User interface and instructions
st.markdown("""
### How to use the Invisibility Cloak:
1. **Position yourself** in front of the camera
2. **Stand completely still** for about 3-4 seconds to let the app capture the background
3. **Put on a blue cloth/shirt** (bright blue works best)
4. **Watch the magic happen!** The blue areas will become invisible

### Tips for best results:
- Use a **bright blue colored cloth** (blue shirt, scarf, or fabric)
- Ensure **good lighting** in the room
- **Stand still** during background capture
- **Avoid blue objects** in the background
""")

# Placeholder for dynamic status updates
status_placeholder = st.empty()

# Initialize video streaming with invisibility cloak processor
webrtc_streamer(
    key="cloak",
    video_processor_factory=CloakProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},  # Video only, no audio needed
    async_processing=True,  # Enable async processing for better performance
)

st.markdown("""
---
### Troubleshooting:
- If the app seems stuck, refresh the page and try again
- Make sure your browser has camera permissions
- For best results, use a desktop/laptop with a good camera
""")
