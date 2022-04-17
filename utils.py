import cv2
import numpy as np
from PIL.Image import Image, BICUBIC


def parse_prediction(y_pred: list, width: int, height: int) -> list[tuple[float, float]]:
    """Parses the raw prediction results

    Args:
        y_pred (list): The normalized predictions from the model
        img_size_x (int): The image width
        img_size_y (int): The image height

    Returns:
        list[tuple[float, float]]: A list of parsed x, y coords for each landmark
    """
    landmarks = []

    for i in range(0, len(y_pred), 2):
        landmark_x, landmark_y = y_pred[i] * width, y_pred[i+1] * height
        landmarks.append((landmark_x, landmark_y))

    return landmarks


def prepare_image(img: np.ndarray) -> np.ndarray:
    """Prepares the image so that it can be given to the model

    Args:
        img (np.ndarray): The original image

    Returns:
        np.ndarray: The model image
    """
    img_temp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_temp = cv2.resize(img_temp, dsize=(
        96, 96), interpolation=cv2.INTER_AREA)
    img_temp = np.expand_dims(img_temp, axis=2)
    img_temp = img_temp / 255
    return img_temp.astype("float32")


def get_glasses_points(landmarks: list) -> dict[str, dict[str, int]]:
    """Gets the wanted landmarks for glasses and labels them

    Args:
        landmarks (list): The parsed landmarks

    Returns:
        dict[str, dict[str, int]]: A dict containing label and its x, y coords
    """
    return {
        "left_eye_center": {
            "x": int(landmarks[0][0]),
            "y": int(landmarks[0][1])
        },
        "right_eye_center": {
            "x": int(landmarks[1][0]),
            "y": int(landmarks[1][1])
        },
        "left_eye_outer": {
            "x": int(landmarks[3][0]),
            "y": int(landmarks[3][1])
        },
        "right_eye_outer": {
            "x": int(landmarks[5][0]),
            "y": int(landmarks[5][1])
        },
    }


def position_glasses(img: Image, glasses_points: dict[str, dict[str, int]]) -> tuple[Image, int, int]:
    """Rotate and get the right position of the glasses image

    Args:
        img (Image): The glasses image
        glasses_points (dict[str, dict[str, int]]): The glasses points

    Returns:
        tuple[Image, int, int]: return rotated image, x and y coords
    """
    width, height = get_glasses_dims(img, glasses_points)

    glasses_resized = img.resize((width, height))

    eye_angle_radians = np.arctan(
        (glasses_points["right_eye_center"]["y"] -
         glasses_points["left_eye_center"]["y"])
        / (glasses_points["left_eye_center"]["x"] - glasses_points["right_eye_center"]["x"]))

    eye_angle_degrees = np.degrees(eye_angle_radians)

    glasses_rotated = glasses_resized.rotate(
        eye_angle_degrees, expand=True, resample=BICUBIC)

    x_offset = int(width * 0.5)
    y_offset = int(height * 0.5)
    pos_x = int((glasses_points["left_eye_center"]["x"] +
                glasses_points["right_eye_center"]["x"]) / 2) - x_offset
    pos_y = int((glasses_points["left_eye_center"]["y"] +
                glasses_points["right_eye_center"]["y"]) / 2) - y_offset

    return glasses_rotated, pos_x, pos_y


def get_glasses_dims(img: Image, points: dict[str, dict[str, int]]) -> tuple[int, int]:
    """Gets the new width and height of the glasses image

    Args:
        img (Image): The classes image
        points (dict[str, dict[str, int]]): The glasses points

    Returns:
        tuple[int, int]: The new width and height of the image
    """
    width = int((points["left_eye_outer"]["x"] -
                points["right_eye_outer"]["x"]) * 1.4)
    height = int(img.size[1] * (width / img.size[0]))

    return width, height


def overlay_transparent(background: list, overlay: list, x: int, y: int) -> list:
    """Overlay an RGBA image over an RGB image

    Args:
        background (list): The image to overlay on
        overlay (list): The image to overlay
        x (int): The x coords of the overlay
        y (int): The y coords of the overlay

    Returns:
        list: The background with the overlay on top of it
    """
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1],
                        1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * \
        background[y:y+h, x:x+w] + mask * overlay_image

    return background
