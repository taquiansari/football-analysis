from ultralytics import YOLO

# Load the Model
model = YOLO('models/best.pt') 

# Run the Model
results = model.predict('input_videos/08fd33_4.mp4', save=True)
# save=True creates a folder to save the video with annonated frames with bounding boxes
print(results[0]) # Printing the result of the first frame of the video
print("=====================================================")

# Looping over the boxes in the first frame
for box in results[0].boxes:
    print(box)

# Each box contains: Coordinates of bounding boxes(Corner method; x1y1 and x2y2), confidence score, detected class label

