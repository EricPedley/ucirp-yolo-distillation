from ultralytics import SAM
import cv2 as cv
import supervision as sv

# Load the model
model = SAM("mobile_sam.pt")

# Predict a segment based on a point prompt
image = cv.imread("rocket.png")
print("predicting")
results = model.predict(image, labels=["dog"])
annotated_frame = image.copy()
for box in results[0].boxes.xyxy[1:]:
    x1,y1,x2,y2 = box.int().cpu().tolist()
    cv.rectangle(annotated_frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv.putText(annotated_frame, "rocket", (x1,y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    

# Visualize the results
cv.imwrite("rocket_annotated.png", annotated_frame)
