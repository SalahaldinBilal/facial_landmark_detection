import numpy as np
import cv2
from PIL.Image import open as openImage
from keras.models import load_model
import time
from utils import parse_prediction, get_glasses_points, overlay_transparent, position_glasses, prepare_image

WIN_NAME = "Coolainator 2000"

cap = cv2.VideoCapture(0)
pTime = 0
model = load_model("assets/model.keras")
glasses_img = openImage("assets/sunglasses.png")
mode = 1

while True:
    success, img = cap.read()
    smaller_size = min(img.shape[0], img.shape[1])
    img = cv2.resize(img, (smaller_size, smaller_size))

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    model_image = prepare_image(img)

    y_pred = model.predict(np.expand_dims(model_image, axis=0))[0]
    orig_size_x, orig_size_y = img.shape[0], img.shape[1]

    landmarks = parse_prediction(y_pred, orig_size_x, orig_size_y)

    if mode == 1:
        glasses_points = get_glasses_points(landmarks)

        positioned_glasses, x, y = position_glasses(
            glasses_img, glasses_points)

        img = overlay_transparent(img, np.array(positioned_glasses), x, y)

    if mode == 2:
        for index, [x, y] in enumerate(landmarks):
            color = (0, 0, 255)
            x = int(x)
            y = int(y)
            img = cv2.drawMarker(img, (x, y), color, thickness=1)
            cv2.putText(img, str(index), (x, y + 5), color=color,
                        thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)

    cv2.putText(img, f'FPS:{int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(WIN_NAME, img)

    key = cv2.waitKey(1)

    if key == ord("1"):
        mode = 1
        glasses_img = openImage("assets/sunglasses.png")
    if key == ord("2"):
        mode = 1
        glasses_img = openImage("assets/glasses.png")
    if key == ord("3"):
        mode = 2

    if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()