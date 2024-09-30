import torch

def classify_image(model, preprocess, image, labels):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_class = torch.max(probabilities, 0)
    class_name = labels[predicted_class.item()]
    return predicted_class.item(), class_name, confidence.item()
