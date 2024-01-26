import cv2
from ultralytics import YOLO
import imageio

# Load the YOLOv8 model
model = YOLO('C:/Users/fstef/Desktop/best (1).pt')

# Open the video file
# Open the video file
video_path = "C:/Users/fstef/Desktop/ch0_strm0_eng0_time2023-12-16_11-15-35.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) * 0.5

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=False, show_labels=False, show_conf=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the frame to the output file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()