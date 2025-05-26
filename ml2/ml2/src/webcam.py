import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from model import CNN
from preprocess import preprocess
import numpy as np

def preprocess_frame(frame, target_size=(50, 50)):
    proc_img = preprocess(frame, target_size=target_size)
    if proc_img is None:
        proc_img = np.zeros((1, target_size[0], target_size[1]), dtype=np.uint8)
    img_pil = Image.fromarray(proc_img[0])
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(img_pil).unsqueeze(0)
    return tensor, proc_img[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_map = {0: "rock", 1: "paper", 2: "scissors"}
    model = CNN(input_size=50, num_classes=3)
    model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    orig_window = "RPS Webcam Test"
    filtered_window = "Filtered"
    cv2.namedWindow(orig_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(filtered_window, cv2.WINDOW_NORMAL)

    last_orig_shape = (640, 480)
    last_filtered_shape = (400, 400)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        input_tensor, filtered_img = preprocess_frame(frame)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = class_map[predicted.item()]

        cv2.putText(frame, f"Prediction: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Get current window sizes
        orig_width = cv2.getWindowImageRect(orig_window)[2]
        orig_height = cv2.getWindowImageRect(orig_window)[3]
        filtered_width = cv2.getWindowImageRect(filtered_window)[2]
        filtered_height = cv2.getWindowImageRect(filtered_window)[3]

        # Fallback/default window sizes
        if orig_width == 0 or orig_height == 0:
            orig_width, orig_height = last_orig_shape
        else:
            last_orig_shape = (orig_width, orig_height)

        if filtered_width == 0 or filtered_height == 0:
            filtered_width, filtered_height = last_filtered_shape
        else:
            last_filtered_shape = (filtered_width, filtered_height)

        # Resize for display
        frame_scaled = cv2.resize(frame, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        filtered_resized = cv2.resize(filtered_img, (filtered_width, filtered_height), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(orig_window, frame_scaled)
        cv2.imshow(filtered_window, filtered_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()