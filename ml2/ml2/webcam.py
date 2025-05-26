import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from train.train_harder import CNN  # Import the model class from your training script

def preprocess_frame(frame, target_size=(50, 50)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    pil_img = Image.fromarray(gray)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(pil_img).unsqueeze(0)  # Add batch dim
    return tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_map = {0: "rock", 1: "paper", 2: "scissors"}

    model = CNN(input_size=50, num_classes=3)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        input_tensor = preprocess_frame(frame).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = class_map[predicted.item()]

        cv2.putText(frame, f"Prediction: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("RPS Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
