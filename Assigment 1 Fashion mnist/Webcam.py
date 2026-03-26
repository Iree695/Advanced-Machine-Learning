import cv2
import numpy as np
import onnxruntime as ort

ONNX_PATH = "fashion_mnist_cnn.onnx"
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) # colour --> gray

    # Central ROI for not using the hole image
    h, w = gray.shape           # height and width of the gray image
    s = min(h, w) // 2          # half of the smaller side
    cx, cy = w // 2, h // 2     # center imagen
    roi = gray[cy - s//2: cy + s//2, cx - s//2: cx + s//2]          # cut a centered square

    img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)   

    # Setting to resemble with the Fashion mnist ( black background, light object)
    img = cv2.GaussianBlur(img, (3, 3), 0)              # Blur to reduce sound
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # auto thereshold(otsu)
    img = 255 - img  # invierte

    x = img.astype(np.float32) / 255.0          # Normalize
    x = x[None, :, :, None]                    # (1, 28, 28, 1)
    return x, roi

def main():
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name      # Real input tensor name

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # open webcam

    if not cap.isOpened():
        raise RuntimeError("No open webcam. Try with index 1 o 2.")

    print("SPACE = capture y clasify | q = exit")

    last_text = "Press SPACE"
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Draw centered ROI
        h, w = frame.shape[:2]
        s = min(h, w) // 2
        # coordenates to the center rectangle
        x1, y1 = w//2 - s//4, h//2 - s//4
        x2, y2 = w//2 + s//4, h//2 + s//4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, last_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"): # "q"
            break

        if k == 32:  # SPACE
            x, roi = preprocess(frame)
            y = sess.run(None, {input_name: x})[0] # (all, dicc, 1ºexit)
            probs = y[0]                    # Vector of 10 values
            pred = int(np.argmax(probs))    # Max => predicted class
            conf = float(probs[pred])       # Trust of the max value
            last_text = f"{CLASSES[pred]} conf={conf:.2f}"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": # Only as a principal model not as a module
    main()

