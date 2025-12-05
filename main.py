import streamlit as st
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


st.set_page_config(page_title="Invisibility Cloak", layout="centered")
st.title("🧙 Invisibility Cloak - Streamlit Edition")


# RTC Configuration (required for Streamlit Cloud)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# -------------------
# Video Processor
# -------------------
class CloakProcessor(VideoProcessorBase):
    def __init__(self):
        self.background = None
        self.frames_captured = 0
        self.background_frames_needed = 40

        # HSV range for BLUE color cloak
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([130, 255, 255])

    def capture_background(self, frame):
        """Accumulate frames for background median estimation."""
        if self.background is None:
            self.background = []

        self.background.append(frame)
        self.frames_captured += 1

        if self.frames_captured == self.background_frames_needed:
            self.background = np.median(self.background, axis=0).astype(np.uint8)
            print("Background captured!")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Capture background for first 40 frames
        if isinstance(self.background, list):
            self.capture_background(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Mask for cloak color
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

        mask_inv = cv2.bitwise_not(mask)

        # Foreground (everything except cloak)
        fg = cv2.bitwise_and(img, img, mask=mask_inv)

        # Background replace
        bg = cv2.bitwise_and(self.background, self.background, mask=mask)

        # Final output
        final = cv2.add(fg, bg)

        return av.VideoFrame.from_ndarray(final, format="bgr24")


# -------------------
# Streamlit UI
# -------------------
st.info("Stand still for 3 seconds so the app can capture your background.")

webrtc_streamer(
    key="cloak",
    video_processor_factory=CloakProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
