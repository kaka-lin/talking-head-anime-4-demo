import os
import sys
import time
import math

import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import scipy.optimize
from scipy.spatial.transform import Rotation

from tha4.charmodel.character_model import CharacterModel
from tha4.image_util import resize_PIL_image, convert_output_image_from_torch_to_numpy


def clamp(x, min_value, max_value):
    return max(min_value, min(max_value, x))


def convert_pose(blendshape_params, euler_angles):
    # Pose parameters
    smile_threshold_min = 0.4
    smile_threshold_max = 0.6
    eye_surprised_max = 0.5
    eye_blink_max = 0.8
    eyebrow_down_max = 0.4
    cheek_squint_min = 0.1
    cheek_squint_max = 0.7
    eye_rotation_factor = 1.0 / 0.75
    jaw_open_min = 0.1
    jaw_open_max = 0.4
    mouth_frown_max = 0.6
    mouth_funnel_min = 0.25
    mouth_funnel_max = 0.5
    iris_small_left = 0.0
    iris_small_right = 0.0
    head_x_offset = 0.0
    head_y_offset = 0.0
    head_z_offset = 0.0
    pose = [0.0 for i in range(45)]

    # Smile
    smile_value = \
        (blendshape_params['mouthSmileLeft'] + blendshape_params['mouthSmileRight']) / 2.0 \
        + blendshape_params['mouthShrugUpper']
    if smile_threshold_min >= smile_threshold_max:
        smile_degree = 0.0
    else:
        if smile_value < smile_threshold_min:
            smile_degree = 0.0
        elif smile_value > smile_threshold_max:
            smile_degree = 1.0
        else:
            smile_degree = (smile_value - smile_threshold_min) / (smile_threshold_max - smile_threshold_min)

    # Eyebrow
    if True:
        brow_inner_up = blendshape_params['browInnerUp']
        brow_outer_up_right = blendshape_params['browOuterUpRight']
        brow_outer_up_left = blendshape_params['browOuterUpLeft']

        brow_up_left = clamp(brow_inner_up + brow_outer_up_left, 0.0, 1.0)
        brow_up_right = clamp(brow_inner_up + brow_outer_up_right, 0.0, 1.0)
    
        pose[6] = brow_up_left
        pose[7] = brow_up_right

        if eyebrow_down_max <= 0.0:
            brow_down_left = 0.0
            brow_down_right = 0.0
        else:
            brow_down_left = (1.0 - smile_degree) \
                             * clamp(blendshape_params['browDownLeft'] / eyebrow_down_max, 0.0, 1.0)
            brow_down_right = (1.0 - smile_degree) \
                              * clamp(blendshape_params['browDownRight'] / eyebrow_down_max, 0.0, 1.0)

        # EyebrowDownMode.TROUBLED:
        pose[0] = brow_down_left
        pose[1] = brow_down_right

        # eyebrow_happy_left
        brow_happy_value = clamp(smile_value, 0.0, 1.0) * smile_degree
        pose[8] = brow_happy_value
        pose[9] = brow_happy_value

    # Eye 
    if True:
        # Surprised: eye_surprised_left_index
        if eye_surprised_max <= 0.0:
            pose[16] = 0.0
            pose[17] = 0.0
        else:
            pose[16] = clamp(blendshape_params['eyeWideLeft'] / eye_surprised_max, 0.0, 1.0)
            pose[17] = clamp(blendshape_params['eyeWideRight'] / eye_surprised_max, 0.0, 1.0)

        # Wink
        if eye_blink_max <= 0.0:
            pose[12] = 0.0
            pose[13] = 0.0
            pose[14] = 0.0
            pose[15] = 0.0
        else:
            pose[12] = (1.0 - smile_degree) * clamp(
                blendshape_params['eyeBlinkLeft'] / eye_blink_max, 0.0, 1.0)
            pose[13] = (1.0 - smile_degree) * clamp(
                blendshape_params['eyeBlinkRight'] / eye_blink_max, 0.0, 1.0)
            pose[14] = smile_degree * clamp(
                blendshape_params['eyeBlinkLeft'] / eye_blink_max, 0.0, 1.0)
            pose[15] = smile_degree * clamp(
                blendshape_params['eyeBlinkRight'] / eye_blink_max, 0.0, 1.0)

        # Lower eyelid
        cheek_squint_denom = cheek_squint_max - cheek_squint_min
        if cheek_squint_denom <= 0.0:
            pose[22] = 0.0
            pose[23] = 0.0
        else:
            pose[22] = clamp((blendshape_params['cheekSquintLeft'] - cheek_squint_min) / cheek_squint_denom, 0.0, 1.0)
            pose[23] = clamp((blendshape_params['cheekSquintRight'] - cheek_squint_min) / cheek_squint_denom, 0.0, 1.0)

    if True:
        # Iris rotation
        eye_rotation_y = (blendshape_params['eyeLookInLeft']
                          - blendshape_params['eyeLookOutLeft']
                          - blendshape_params['eyeLookInRight']
                          + blendshape_params['eyeLookOutRight']) / 2.0 * eye_rotation_factor
        pose[38] = clamp(eye_rotation_y, -1.0, 1.0)

        eye_rotation_x = (blendshape_params['eyeLookUpLeft']
                          + blendshape_params['eyeLookUpRight']
                          - blendshape_params['eyeLookDownLeft']
                          - blendshape_params['eyeLookDownRight']) / 2.0 * eye_rotation_factor
        pose[37] = clamp(eye_rotation_x, -1.0, 1.0)

    # Iris size
    if True:
        pose[24] = iris_small_left
        pose[25] = iris_small_right

    # Head rotation
    if True:
        euler_angles[0] -= head_x_offset
        euler_angles[1] -= head_y_offset
        euler_angles[2] -= head_z_offset

        x_param = clamp(-euler_angles[0] * 180.0 / math.pi, -15.0, 15.0) / 15.0
        pose[39] = x_param

        y_param = clamp(-euler_angles[1] * 180.0 / math.pi, -10.0, 10.0) / 10.0
        pose[40] = y_param
        pose[42] = y_param

        z_param = clamp(euler_angles[2] * 180.0 / math.pi, -15.0, 15.0) / 15.0
        pose[41] = z_param
        pose[43] = z_param

    # Mouth
    if True:
        jaw_open_denom = jaw_open_max - jaw_open_min
        if jaw_open_denom <= 0:
            mouth_open = 0.0
        else:
            mouth_open = clamp((blendshape_params['jawOpen'] - jaw_open_min) / jaw_open_denom, 0.0, 1.0)

        pose[26] = mouth_open
        pose[34] = clamp(smile_value, 0.0, 1.0)
        pose[35] = clamp(smile_value, 0.0, 1.0)

        is_mouth_open = mouth_open > 0.0
        if not is_mouth_open:
            if mouth_frown_max <= 0:
                mouth_frown_value = 0.0
            else:
                mouth_frown_value = clamp((blendshape_params['mouthFrownLeft'] 
                                           + blendshape_params['mouthFrownRight']) / mouth_frown_max, 0.0, 1.0)
        
            pose[32] = mouth_frown_value
            pose[33] = mouth_frown_value
        else:
            mouth_lower_down = clamp(blendshape_params['mouthLowerDownLeft'] 
                                     + blendshape_params['mouthLowerDownRight'], 0.0, 1.0)
            mouth_funnel = blendshape_params['mouthFunnel']
            mouth_pucker = blendshape_params['mouthPucker']

            mouth_point = [mouth_open, mouth_lower_down, mouth_funnel, mouth_pucker]

            aaa_point = [1.0, 1.0, 0.0, 0.0]
            iii_point = [0.0, 1.0, 0.0, 0.0]
            uuu_point = [0.5, 0.3, 0.25, 0.75]
            ooo_point = [1.0, 0.5, 0.5, 0.4]

            decomp = np.array([0, 0, 0, 0])
            M = np.array([
                aaa_point,
                iii_point,
                uuu_point,
                ooo_point
            ])

            def loss(decomp):
                return np.linalg.norm(np.matmul(decomp, M) - mouth_point) \
                       + 0.01 * np.linalg.norm(decomp, ord=1)
        
            opt_result = scipy.optimize.minimize(
                loss, decomp, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])

            decomp = opt_result["x"]
            restricted_decomp = [decomp.item(0), decomp.item(1), decomp.item(2), decomp.item(3)]
            pose[26] = restricted_decomp[0]
            pose[27] = restricted_decomp[1]
            mouth_funnel_denom = mouth_funnel_max - mouth_funnel_min
            if mouth_funnel_denom <= 0:
                ooo_alpha = 0.0
                uo_value = 0.0
            else:
                ooo_alpha = clamp((mouth_funnel - mouth_funnel_min) / mouth_funnel_denom, 0.0, 1.0)
                uo_value = clamp(restricted_decomp[2] + restricted_decomp[3], 0.0, 1.0)
            pose[28] = uo_value * (1.0 - ooo_alpha)
            pose[30] = uo_value * ooo_alpha

    # breathing frequency
    value = 0.0
    pose[44] = value

    return pose


if __name__ == "__main__":
    # STEP 0: config for CUDA or MPS (Metal GPU)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        # model not supported on MPS,
        # so we use CPU
        device = "cpu"
    
    character_model_file_name = "data/character_models/lambda_01/character_model.yaml"

    # Load the character model that you want
    character_model = CharacterModel.load(character_model_file_name)
    # Get the face and body model
    poser = character_model.get_poser(device)

    # Load the character model that you want
    character_model = CharacterModel.load(character_model_file_name)
    # Get the face and body model
    poser = character_model.get_poser(device)

    # Get input image
    torch_source_image = character_model.get_character_image(device)

    # STEP 1: Create an FaceLandmarker object.
    base_options = python.BaseOptions(
        model_asset_path='data/thirdparty/mediapipe/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # STEP 2: Load the image or Open VideoCapture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not open!!!")
        sys.exit()
    
    while True:
        ret, frame = cap.read()
       
        if not ret:
            print("Can't receive frame (stream end?)")
            continue
        
        # STEP 3: Convert the image to MediaPipe Image
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_image)
        
        # STEP 4: Detect the face landmarks
        time_ms = int(time.time() * 1000)
        detection_result = face_landmarker.detect_for_video(mediapipe_image, time_ms)
        
        if len(detection_result.facial_transformation_matrixes) > 0:
            xform_matrix = detection_result.facial_transformation_matrixes[0]
            blendshape_params = {}
            for item in detection_result.face_blendshapes[0]:
                blendshape_params[item.category_name] = item.score
            M = xform_matrix[0:3, 0:3]
            rot = Rotation.from_matrix(M)
            euler_angles = rot.as_euler('xyz', degrees=True)

            current_pose = convert_pose(blendshape_params, euler_angles)
            pose = torch.tensor(current_pose, dtype=torch.float32)

            # Inference
            with torch.no_grad():
                output_image = poser.pose(torch_source_image, pose)[0].float()
            numpy_image = convert_output_image_from_torch_to_numpy(output_image)

            cv2.imshow("THA4 Demo", cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
