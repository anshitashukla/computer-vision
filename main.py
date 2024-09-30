import cv2
import matplotlib.pyplot as plt
from model import get_model
from image_processing import apply_pixelation
from classification import classify_image

if __name__ == "__main__":
    model, labels, preprocess = get_model()
    
    image_path = "data/testImage1.jpg"
    img = cv2.imread(image_path)
    
    pixelated_img = apply_pixelation(img)
    predicted_class, class_name, confidence = classify_image(model, preprocess, pixelated_img, labels)
    print(f"Predicted class: {predicted_class}, Class name: {class_name}, Confidence: {confidence:.4f}")
    plt.imshow(cv2.cvtColor(pixelated_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Class: {class_name}, Confidence: {confidence:.4f}")
    plt.show()
