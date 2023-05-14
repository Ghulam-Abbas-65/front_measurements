import mediapipe as mp
import cv2
import math
import numpy as np
import flask
import io
# import json
# import base64
from PIL import Image
from flask import Flask, jsonify, request, Response, send_file


def prepare_image(img):
    img = Image.open(io.BytesIO(img)).convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1].copy()
    return img


def move_right_extreme(img, point):
    # _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
    pixel = (0, 0, 0)
    while pixel < (240, 240, 240):
        pixel = img.getpixel((point))
        # pixel = img[point[1], point[0]]
        point = (point[0] - 1, point[1])
    return point


def move_left_extreme(img, point):
    # _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
    pixel = (0, 0, 0)
    while pixel < (240, 240, 240):
        pixel = img.getpixel((point))
        # pixel = img[point[1], point[0]]
        point = (point[0] + 1, point[1])
    return point


def move_up_extreme(img, point):
    # _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
    pixel = (0, 0, 0)
    while pixel < (240, 240, 240):
        # while (pixel.any() < 220):
        pixel = img.getpixel((point))
        # pixel = img[int(point[1]), int(point[0])]
        point = (point[0], point[1] - 1)
    return point


def move_down_extreme(img, point):
    # _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
    pixel = (0, 0, 0)
    while pixel < (240, 240, 240):
        pixel = img.getpixel((point))
        # pixel = img[point[1], point[0]]
        point = (point[0], point[1] + 1)
    return point


def get_total_height(left_eye_inner, right_eye_inner, left_foot_index, image):
    eye_point = (
        (left_eye_inner[0] + right_eye_inner[0]) / 2, left_eye_inner[1])
    top_point = np.array(move_up_extreme(image, eye_point))
    bottom_point = np.array((eye_point[0], left_foot_index[1]))
    height = np.linalg.norm(top_point - bottom_point)
    return height, (top_point, bottom_point)


def get_shoulder(left_shoulder, right_shoulder, factor, image, change_factor, mode='shoulder'):
    if mode == 'chest':
        left_chest = (
            left_shoulder[0], left_shoulder[1] + factor * 2.54 * change_factor)
        right_chest = (
            right_shoulder[0], right_shoulder[1] + factor * 2.54 * change_factor)
    else:
        left_chest = left_shoulder
        right_chest = right_shoulder

    left_point = np.array(move_up_extreme(
        image, move_left_extreme(image, left_chest)))
    right_point = np.array(move_up_extreme(
        image, move_right_extreme(image, right_chest)))
    shoulder_length = np.linalg.norm(left_point - right_point) / factor
    return shoulder_length, (left_point, right_point)


def get_arm_length(shoulder_point, elbow_point, wrist_point, factor, image):
    shoulder = np.array(move_up_extreme(image, shoulder_point))
    # shoulder = np.array(shoulder_point)
    elbow = np.array(elbow_point)
    wrist = np.array(wrist_point)
    shoulder_elbow = np.linalg.norm(shoulder - elbow)
    elbow_wrist = np.linalg.norm(wrist - elbow)
    arm_length = (shoulder_elbow + elbow_wrist) / factor
    return arm_length, (shoulder, wrist)

# def get_arm_length(shoulder_point, wrist_point, factor, image):
#     shoulder = np.array(move_up_extreme(
#         image, move_left_extreme(image, shoulder_point)))
#     wrist = np.array(wrist_point)
#     shoulder_elbow = np.linalg.norm(shoulder - wrist)
#     return shoulder_elbow, (shoulder, wrist)


def get_shoulder_to_waist(shoulder_point, waist_point, factor, image):
    shoulder = np.array(move_up_extreme(image, shoulder_point))
    waist = np.array((shoulder[0], waist_point[1]))
    shoulder_waist_length = np.linalg.norm(shoulder - waist) / factor
    return shoulder_waist_length, (shoulder, waist)


def get_leg_length(waist_point, knee_point, heel_point, factor, image):
    waist = np.array(waist_point)
    knee = np.array(knee_point)
    heel = np.array(heel_point)
    waist_knee = np.linalg.norm(waist - knee)
    knee_heel = np.linalg.norm(knee - heel)
    leg_length = (waist_knee + knee_heel) / factor
    return leg_length, (waist, heel)


# def get_hip(left_hip, right_hip, factor, image):
#     left_point = np.array(move_left_extreme(image, left_hip))
#     right_point = np.array(move_right_extreme(image, right_hip))
#     hip_length = np.linalg.norm(left_point - right_point) / factor
#     return hip_length, (left_point, right_point)


# def get_thigh(hip_point, knee_point, factor, image):
#     thigh_point = (hip_point[0], (hip_point[1] + knee_point[1]) / 2)
#     thigh_left = np.array(move_left_extreme(image, thigh_point))
#     thigh_right = np.array(move_right_extreme(image, thigh_point))
#     thigh_length = np.linalg.norm(
#         thigh_left - thigh_right) / factor
#     return thigh_length, (thigh_left, thigh_right)


# def get_wrist_length(wrist_point, factor, image):
#     left_point = np.array(move_left_extreme(image, wrist_point))
#     right_point = np.array(move_right_extreme(image, wrist_point))
#     wrist_length = np.linalg.norm(left_point - right_point) / factor
#     return wrist_length, (left_point, right_point)


# def get_bicep_length(shoulder_point, elbow_point, factor, image):
#     shoulder_point = move_left_extreme(image, shoulder_point)
#     bicep_point = ((shoulder_point[0] + elbow_point[0]) / 2,
#                    (shoulder_point[1] + elbow_point[1]) / 2)
#     left_point = np.array(move_left_extreme(image, bicep_point))
#     right_point = np.array(move_right_extreme(image, bicep_point))
#     bicep_length = np.linalg.norm(left_point - right_point) / factor
#     return bicep_length, (left_point, right_point)


# def get_neck(left_shoulder, right_shoulder, factor, image):
#     left_shoulder_extreme = move_up_extreme(
#         image, move_left_extreme(image, left_shoulder))
#     right_shoulder_extreme = move_up_extreme(
#         image, move_right_extreme(image, right_shoulder))

#     neck_point = (
#         (left_shoulder_extreme[0] + right_shoulder[0]) / 2, right_shoulder_extreme[1] - factor * 2.54 * 5)
#     left_point = np.array(move_left_extreme(image, neck_point))
#     right_point = np.array(move_right_extreme(image, neck_point))
#     neck_length = np.linalg.norm(left_point - right_point) / factor
#     return neck_length, (left_point, right_point)


# def get_knee_width(knee_point, factor, image):
#     left_point = np.array(move_left_extreme(image, knee_point))
#     right_point = np.array(move_right_extreme(image, knee_point))
#     knee_length = np.linalg.norm(
#         left_point - right_point) / factor
#     return knee_length, (left_point, right_point)


def get_waist(left_hip, right_hip, factor, image, hip_waist_factor):
    left_point = (left_hip[0], left_hip[1] - factor * 2.54 * hip_waist_factor)
    right_point = (right_hip[0], right_hip[1] -
                   factor * 2.54 * hip_waist_factor)
    left_point_waist = np.array(move_left_extreme(image, left_point))
    right_point_waist = np.array(move_right_extreme(image, right_point))
    waist_length = np.linalg.norm(
        left_point_waist - right_point_waist) / factor
    return waist_length, (left_point_waist, right_point_waist)


def get_crotch(left_hip, right_hip, factor, image):
    waist_point = ((left_hip[0] + right_hip[0]) / 2, left_hip[1])
    crotch_point = np.array(move_down_extreme(image, waist_point))
    crotch_length = np.linalg.norm(
        np.array(waist_point) - crotch_point) / factor
    return crotch_length, (waist_point, crotch_point)

# def get_side_length(point, factor, image, change_factor, mode="normal"):
#     if mode == 'chest':
#         left_point = (point[0], point[1] + factor * 2.54 * change_factor)
#     else:
#         left_point = (point[0], point[1] - factor * 2.54 * change_factor)
#     left_point_extreme = np.array(move_left_extreme(image, left_point))
#     right_point_extreme = np.array(move_right_extreme(image, left_point))
#     side_length = np.linalg.norm(
#         left_point_extreme - right_point_extreme) / factor
#     return side_length, (left_point_extreme, right_point_extreme)


# def get_girth(a, b):
#     circumference = math.pi * (a + b) * ((3 * (a - b) ** 2) / ((a + b) **
#                                                                2 * (math.sqrt(
#                 (-3 * (a - b) ** 2) / (a + b) ** 2 + 4) + 10)) + 1)
#     return circumference


def get_key_points(img):
    images = {'name': img}

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Run MediaPipe Pose with `enable_segmentation=True` to get pose segmentation.
    # with mp_pose.Pose(
    #         static_image_mode=True, min_detection_confidence=0.9,
    #         model_complexity=2, enable_segmentation=True) as pose:
    #     for name, image in images.items():
    #         results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #         # Draw pose segmentation.
    #         annotated_image = image.copy()
    #         red_img = np.zeros_like(annotated_image, dtype=np.uint8)
    #         red_img[:, :] = (255, 255, 255)
    #         segm_2class = 0.2 + 0.8 * results.segmentation_mask
    #         segm_2class = np.repeat(segm_2class[..., np.newaxis], 3, axis=2)
    #         annotated_image = annotated_image * \
    #             segm_2class + red_img * (1 - segm_2class)
    #         # cv2.imwrite("savedImage.jpg", annotated_image)
    #         im = Image.fromarray(
    #             (annotated_image * 1).astype(np.uint8)).convert('RGB')

    from typing_extensions import Annotated
    # Run MediaPipe Pose and draw pose landmarks.
    with mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.9, model_complexity=2) as pose:
        for name, image in images.items():
            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print nose landmark.
            image_hight, image_width, _ = image.shape
            if not results.pose_landmarks:
                continue

            poseLandmarks = results.pose_landmarks.landmark
            landmarks = mp_pose.PoseLandmark

            left_eye_inner = (round(poseLandmarks[landmarks.LEFT_EYE_INNER].x * image_width),
                              round(poseLandmarks[landmarks.LEFT_EYE_INNER].y * image_hight))
            left_eye = (round(poseLandmarks[landmarks.LEFT_EYE].x * image_width),
                        round(poseLandmarks[landmarks.LEFT_EYE].y * image_hight))
            left_eye_outer = (round(poseLandmarks[landmarks.LEFT_EYE_OUTER].x * image_width),
                              round(poseLandmarks[landmarks.LEFT_EYE_OUTER].y * image_hight))

            right_eye_inner = (round(poseLandmarks[landmarks.RIGHT_EYE_INNER].x * image_width),
                               round(poseLandmarks[landmarks.RIGHT_EYE_INNER].y * image_hight))
            right_eye = (round(poseLandmarks[landmarks.RIGHT_EYE].x * image_width),
                         round(poseLandmarks[landmarks.RIGHT_EYE].y * image_hight))
            right_eye_outer = (round(poseLandmarks[landmarks.RIGHT_EYE_OUTER].x * image_width),
                               round(poseLandmarks[landmarks.RIGHT_EYE_OUTER].y * image_hight))

            left_shoulder = (round(poseLandmarks[landmarks.LEFT_SHOULDER].x * image_width),
                             round(poseLandmarks[landmarks.LEFT_SHOULDER].y * image_hight))
            right_shoulder = (round(poseLandmarks[landmarks.RIGHT_SHOULDER].x * image_width),
                              round(poseLandmarks[landmarks.RIGHT_SHOULDER].y * image_hight))

            left_elbow = (round(poseLandmarks[landmarks.LEFT_ELBOW].x * image_width),
                          round(poseLandmarks[landmarks.LEFT_ELBOW].y * image_hight))
            right_elbow = (round(poseLandmarks[landmarks.RIGHT_ELBOW].x * image_width),
                           round(poseLandmarks[landmarks.RIGHT_ELBOW].y * image_hight))

            left_wrist = (round(poseLandmarks[landmarks.LEFT_WRIST].x * image_width),
                          round(poseLandmarks[landmarks.LEFT_WRIST].y * image_hight))
            right_wrist = (round(poseLandmarks[landmarks.RIGHT_WRIST].x * image_width),
                           round(poseLandmarks[landmarks.RIGHT_WRIST].y * image_hight))

            left_hip = (round(poseLandmarks[landmarks.LEFT_HIP].x * image_width),
                        round(poseLandmarks[landmarks.LEFT_HIP].y * image_hight))
            right_hip = (round(poseLandmarks[landmarks.RIGHT_HIP].x * image_width),
                         round(poseLandmarks[landmarks.RIGHT_HIP].y * image_hight))

            left_knee = (round(poseLandmarks[landmarks.LEFT_KNEE].x * image_width),
                         round(poseLandmarks[landmarks.LEFT_KNEE].y * image_hight))
            right_knee = (round(poseLandmarks[landmarks.RIGHT_KNEE].x * image_width),
                          round(poseLandmarks[landmarks.RIGHT_KNEE].y * image_hight))

            left_ankle = (round(poseLandmarks[landmarks.LEFT_ANKLE].x * image_width),
                          round(poseLandmarks[landmarks.LEFT_ANKLE].y * image_hight))
            right_ankle = (round(poseLandmarks[landmarks.RIGHT_ANKLE].x * image_width),
                           round(poseLandmarks[landmarks.RIGHT_ANKLE].y * image_hight))

            left_heel = (round(poseLandmarks[landmarks.LEFT_HEEL].x * image_width),
                         round(poseLandmarks[landmarks.LEFT_HEEL].y * image_hight))
            right_heel = (round(poseLandmarks[landmarks.RIGHT_HEEL].x * image_width),
                          round(poseLandmarks[landmarks.RIGHT_HEEL].y * image_hight))

            left_foot_index = (round(poseLandmarks[landmarks.LEFT_FOOT_INDEX].x * image_width),
                               round(poseLandmarks[landmarks.LEFT_FOOT_INDEX].y * image_hight))
            right_foot_index = (round(poseLandmarks[landmarks.RIGHT_FOOT_INDEX].x * image_width),
                                round(poseLandmarks[landmarks.RIGHT_FOOT_INDEX].y * image_hight))

            left_thumb = (round(poseLandmarks[landmarks.LEFT_THUMB].x * image_width),
                          round(poseLandmarks[landmarks.LEFT_THUMB].y * image_hight))
            right_thumb = (round(poseLandmarks[landmarks.RIGHT_THUMB].x * image_width),
                           round(poseLandmarks[landmarks.RIGHT_THUMB].y * image_hight))

            left_ear = (round(poseLandmarks[landmarks.LEFT_EAR].x * image_width),
                        round(poseLandmarks[landmarks.LEFT_EAR].y * image_hight))
            right_ear = (round(poseLandmarks[landmarks.RIGHT_EAR].x * image_width),
                         round(poseLandmarks[landmarks.RIGHT_EAR].y * image_hight))

    key_points = {"left_eye_inner": left_eye_inner, "left_eye": left_eye, "left_eye_outer": left_eye_outer,
                  "right_eye_inner": right_eye_inner, "right_eye": right_eye, "right_eye_outer": right_eye_outer,
                  "left_shoulder": left_shoulder, "right_shoulder": right_shoulder, "left_elbow": left_elbow,
                  "right_elbow": right_elbow, "left_wrist": left_wrist, "right_wrist": right_wrist,
                  "left_hip": left_hip, "right_hip": right_hip, "left_knee": left_knee, "right_knee": right_knee,
                  "left_ankle": left_ankle, "right_ankle": right_ankle, "left_heel": left_heel,
                  "right_heel": right_heel, "left_foot_index": left_foot_index, "right_foot_index": right_foot_index,
                  'left_ear': left_ear, 'right_ear': right_ear, 'left_thumb': left_thumb, 'right_thumb': right_thumb}

    # return key_points, im
    return key_points


def get_front_measurements(img, original_height):
    _, bn_img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
    im = Image.fromarray(bn_img)
    # key_points, im = get_key_points(img)
    key_points = get_key_points(img)

    extended_points = list()
    factor, height_points = get_total_height(
        key_points["left_eye_inner"], key_points["right_eye_inner"], key_points["left_heel"], im)
    factor = factor / original_height
    extended_points.append(height_points)

    shoulder, shoulder_points = get_shoulder(
        key_points["left_shoulder"], key_points["right_shoulder"], factor, im, 1)
    extended_points.append(shoulder_points)

    # chest, chest_points = get_shoulder(key_points['left_shoulder'], key_points['right_shoulder'], factor, im, 6,
    #                                    mode='chest')
    # extended_points.append(chest_points)

    arm, arm_points = get_arm_length(key_points["left_shoulder"], key_points["left_elbow"],
                                     key_points["left_wrist"], factor, im)
    # arm, arm_points = get_arm_length(key_points["left_shoulder"],
    #                                  key_points["left_wrist"], factor, im)
    extended_points.append(arm_points)

    shoulder_to_hip, shoulder_hip_points = get_shoulder_to_waist(
        key_points["left_shoulder"], key_points["left_hip"], factor, im)
    extended_points.append(shoulder_hip_points)

    # hip, hip_points = get_hip(key_points["left_hip"],
    #                           key_points["right_hip"], factor, im)
    # extended_points.append(hip_points)

    # upper_waist, upper_waist_points = get_waist(key_points["left_hip"],
    #                                             key_points["right_hip"], factor, im, 6.67)
    # extended_points.append(upper_waist_points)

    waist, waist_points = get_waist(key_points["left_hip"],
                                    key_points["right_hip"], factor, im, 5)
    extended_points.append(waist_points)

    crotch, crotch_points = get_crotch(
        key_points["left_hip"], key_points["right_hip"], factor, im)

    extended_points.append(crotch_points)

    # lower_waist, lower_waist_points = get_waist(key_points["left_hip"],
    #                                             key_points["right_hip"], factor, im, 3)
    # extended_points.append(lower_waist_points)

    leg, leg_points = get_leg_length(waist_points[0], key_points["left_knee"],
                                     key_points["left_heel"], factor, im)
    extended_points.append(leg_points)

    inside_leg_length = leg - crotch

    half_back = shoulder / 2

    # knee, knee_points = get_knee_width(key_points['left_knee'], factor, im)
    # extended_points.append(knee_points)

    # bicep, bicep_points = get_bicep_length(
    #     key_points["left_shoulder"], key_points["left_elbow"], factor, im)
    # extended_points.append(bicep_points)

    # wirst, wrist_points = get_wrist_length(
    #     key_points["left_wrist"], factor, im)
    # extended_points.append(wrist_points)

    # thigh, thigh_points = get_thigh(
    #     key_points["left_hip"], key_points["left_knee"], factor, im)
    # extended_points.append(thigh_points)

    # neck, neck_points = get_neck(key_points["left_shoulder"],
    #                              key_points["right_shoulder"], factor, im)
    # extended_points.append(neck_points)

    # ankle, ankle_points = get_knee_width(key_points['left_ankle'], factor, im)
    # extended_points.append(ankle_points)

    # print(extended_points)

    for point in extended_points:
        cv2.line(img, (int(point[0][0]), int(point[0][1])), (int(point[1][0]), int(point[1][1])), color=(255, 0, 0),
                 thickness=2)
    for point in extended_points:
        cv2.line(bn_img, (int(point[0][0]), int(point[0][1])), (int(point[1][0]), int(point[1][1])), color=(255, 0, 0),
                 thickness=2)

    measurements = {"shoulder": shoulder / 2.54, "arm": (arm / 2.54),
                    "shoulder_to_hip_length": shoulder_to_hip / 2.54, "leg": leg / 2.54, 'crotch': crotch / 2.54, 'inside_leg_length': inside_leg_length / 2.54, 'half_back': half_back / 2.54}

    return measurements, img, bn_img


# def get_side_measurements(img, original_height):
#     key_points, im = get_key_points(img)
#     extended_points = list()
#     factor, height_points = get_total_height(
#         key_points["left_ear"], key_points["right_ear"], key_points["left_foot_index"], im)
#     factor = factor / original_height
#     extended_points.append(height_points)

#     chest, chest_points = get_side_length(
#         key_points["left_shoulder"], factor, im, 6.67, mode='chest')
#     extended_points.append(chest_points)

#     upper_waist, upper_waist_points = get_side_length(key_points["left_hip"],
#                                                       factor, im, 6.67)
#     extended_points.append(upper_waist_points)

#     waist, waist_points = get_side_length(key_points["left_hip"],
#                                           factor, im, 5)
#     extended_points.append(waist_points)

#     lower_waist, lower_waist_points = get_side_length(key_points["left_hip"],
#                                                       factor, im, 3)
#     extended_points.append(lower_waist_points)

#     hip, hip_points = get_side_length(key_points["left_hip"],
#                                       factor, im, 1)
#     extended_points.append(hip_points)

#     knee, knee_points = get_side_length(
#         key_points['left_knee'], factor, im, 1)
#     extended_points.append(knee_points)

#     thigh, thigh_points = get_thigh(
#         key_points["left_hip"], key_points["left_knee"], factor, im)
#     extended_points.append(thigh_points)

#     neck, neck_points = get_neck(key_points["left_shoulder"],
#                                  key_points["right_shoulder"], factor, im)
#     extended_points.append(neck_points)

#     ankle, ankle_points = get_knee_width(key_points['left_ankle'], factor, im)
#     extended_points.append(ankle_points)

#     for point in extended_points:
#         for i in point:
#             cv2.circle(img, (int(i[0]), int(i[1])), 0, (0, 0, 0), 5)

#     print(extended_points)

#     for point in extended_points:
#         cv2.line(img, (int(point[0][0]), int(point[0][1])), (int(point[1][0]), int(point[1][1])), color=(255, 0, 0),
#                  thickness=2)

#     measurements = {'chest': chest / 2.54, "upper_waist": upper_waist / 2.54, "waist": waist / 2.54,
#                     "lower_waist": lower_waist / 2.54, "hip": hip / 2.54, "knee": knee / 2.54, "thigh": thigh / 2.54,
#                     "neck": neck / 2.54, 'ankle': ankle / 2.54}

#     return measurements, img


# def get_measurements(front_image, side_image, original_height):
#     # front_image = front_image - background_image
#     # side_image = side_image - background_image
#     front_measurements, front_img = get_front_measurements(
#         front_image, original_height)
#     side_measurements, side_img = get_side_measurements(
#         side_image, original_height)

#     chest = get_girth(front_measurements['chest'] /
#                       2, side_measurements['chest'] / 2)

#     upper_waist = get_girth(front_measurements['upper_waist'] /
#                             2, side_measurements['upper_waist'] / 2)

#     waist = get_girth(front_measurements['waist'] /
#                       2, side_measurements['waist'] / 2)

#     lower_waist = get_girth(front_measurements['lower_waist'] /
#                             2, side_measurements['lower_waist'] / 2)

#     hip = get_girth(front_measurements['hip'] /
#                     2, side_measurements['hip'] / 2)

#     thigh = get_girth(
#         front_measurements['thigh'] / 2, side_measurements['thigh'] / 2)
#     knee = get_girth(
#         front_measurements['knee'] / 2, side_measurements['knee'] / 2)

#     ankle = get_girth(
#         front_measurements['ankle'] / 2, side_measurements['ankle'] / 2)

#     neck = get_girth(
#         front_measurements['neck'] / 2, side_measurements['neck'] / 2)

#     measurements = {"shoulder": front_measurements['shoulder'], 'chest': chest, "arm": front_measurements['arm'],
#                     "shoulder to hip": front_measurements['shoulder to hip length'], 'upper waist': upper_waist,
#                     "waist": waist, "lower_waist": lower_waist, 'hip': hip, "leg": front_measurements['leg'],
#                     "knee girth": knee, "bicep girth": front_measurements['bicep'],
#                     "wrist": front_measurements['wrist'], "thigh girth": thigh, "neck": neck, 'waist': waist,
#                     "ankle": ankle}

#     return measurements, front_img, side_img


app = Flask(__name__)


@app.route('/', methods=['POST'])
def infer_image():
    if 'front' not in request.files:
        return "Please try again. The Image doesn't exist"

    front = request.files.get('front')
    # side = request.files.get('side')
    if not front:
        return "No Front Image"
    # elif not side:
    #     return "No Side Image"

    height = request.form.get('height', default=165, type=int)

    front_img_bytes = front.read()
    # side_img_bytes = side.read()

    front_img = prepare_image(front_img_bytes)
    # side_img = prepare_image(side_img_bytes)

    # measurements, front_image, side_image = get_measurements(
    #     front_img, side_img, height)
    # measurements, front_image = get_front_measurements(front_img, height)
    measurements, front_image, bn_img = get_front_measurements(
        front_img, height)
    cv2.imwrite('file.jpg', front_image)
    # cv2.imwrite('mask.jpg', bn_img)
#     def generate():
#         return int.from_bytes(b'\x00\x10', byteorder = 'big')
# # Create a Flask response object
#     response = Response()

#     # Set the mimetype to multipart/x-mixed-replace to send multiple images
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.mimetype = 'image/jpeg'
#     response.data = bytes(generate()).decode('utf-8')
#     return response

    return jsonify(measurements)
    # return send_file(front_image, mimetype='image/jpeg')


if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0')
    app.run()
