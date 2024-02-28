import math
import time
from enum import Enum
from typing import Optional, List, Callable

import numpy
import scipy.optimize
import wx
from scipy.spatial.transform import Rotation

from tha4.poser.modes.pose_parameters import get_pose_parameters
from tha4.mocap.mediapipe_constants import MOUTH_SMILE_LEFT, MOUTH_SHRUG_UPPER, MOUTH_SMILE_RIGHT, \
    BROW_INNER_UP, BROW_OUTER_UP_RIGHT, BROW_OUTER_UP_LEFT, BROW_DOWN_LEFT, BROW_DOWN_RIGHT, EYE_WIDE_LEFT, \
    EYE_WIDE_RIGHT, EYE_BLINK_LEFT, EYE_BLINK_RIGHT, CHEEK_SQUINT_LEFT, CHEEK_SQUINT_RIGHT, EYE_LOOK_IN_LEFT, \
    EYE_LOOK_OUT_LEFT, EYE_LOOK_IN_RIGHT, EYE_LOOK_OUT_RIGHT, EYE_LOOK_UP_LEFT, EYE_LOOK_UP_RIGHT, EYE_LOOK_DOWN_RIGHT, \
    EYE_LOOK_DOWN_LEFT, JAW_OPEN, MOUTH_FROWN_LEFT, MOUTH_FROWN_RIGHT, \
    MOUTH_LOWER_DOWN_LEFT, MOUTH_LOWER_DOWN_RIGHT, MOUTH_FUNNEL, MOUTH_PUCKER
from tha4.mocap.mediapipe_face_pose import MediaPipeFacePose
from tha4.mocap.mediapipe_face_pose_converter import MediaPipeFacePoseConverter


class EyebrowDownMode(Enum):
    TROUBLED = 1
    ANGRY = 2
    LOWERED = 3
    SERIOUS = 4


class WinkMode(Enum):
    NORMAL = 1
    RELAXED = 2


def rad_to_deg(rad):
    return rad * 180.0 / math.pi


def deg_to_rad(deg):
    return deg * math.pi / 180.0


def clamp(x, min_value, max_value):
    return max(min_value, min(max_value, x))


class MediaPipeFacePoseConverter00Args:
    def __init__(self,
                 smile_threshold_min: float = 0.4,
                 smile_threshold_max: float = 0.6,
                 eyebrow_down_mode: EyebrowDownMode = EyebrowDownMode.ANGRY,
                 wink_mode: WinkMode = WinkMode.NORMAL,
                 eye_surprised_max: float = 0.5,
                 eye_blink_max: float = 0.8,
                 eyebrow_down_max: float = 0.4,
                 cheek_squint_min: float = 0.1,
                 cheek_squint_max: float = 0.7,
                 eye_rotation_factor: float = 1.0 / 0.75,
                 jaw_open_min: float = 0.1,
                 jaw_open_max: float = 0.4,
                 mouth_frown_max: float = 0.6,
                 mouth_funnel_min: float = 0.25,
                 mouth_funnel_max: float = 0.5,
                 iris_small_left=0.0,
                 iris_small_right=0.0,
                 head_x_offset=0.0,
                 head_y_offset=0.0,
                 head_z_offset=0.0):
        self.iris_small_right = iris_small_left
        self.iris_small_left = iris_small_right

        self.wink_mode = wink_mode

        self.mouth_funnel_max = mouth_funnel_max
        self.mouth_funnel_min = mouth_funnel_min
        self.mouth_frown_max = mouth_frown_max

        self.jaw_open_max = jaw_open_max
        self.jaw_open_min = jaw_open_min

        self.eye_rotation_factor = eye_rotation_factor

        self.cheek_squint_max = cheek_squint_max
        self.cheek_squint_min = cheek_squint_min

        self.eyebrow_down_max = eyebrow_down_max

        self.eye_blink_max = eye_blink_max
        self.eye_surprised_max = eye_surprised_max

        self.smile_threshold_min = smile_threshold_min
        self.smile_threshold_max = smile_threshold_max

        self.head_z_offset = head_z_offset
        self.head_y_offset = head_y_offset
        self.head_x_offset = head_x_offset

        self.eyebrow_down_mode = eyebrow_down_mode

    def set_smile_threshold_min(self, new_value: float):
        self.smile_threshold_min = new_value

    def set_smile_threshold_max(self, new_value: float):
        self.smile_threshold_max = new_value

    def set_eye_surprised_max(self, new_value: float):
        self.eye_surprised_max = new_value

    def set_eye_blink_max(self, new_value: float):
        self.eye_blink_max = new_value

    def set_eyebrow_down_max(self, new_value: float):
        self.eyebrow_down_max = new_value

    def set_cheek_squint_min(self, new_value: float):
        self.cheek_squint_min = new_value

    def set_cheek_squint_max(self, new_value: float):
        self.cheek_squint_max = new_value

    def set_jaw_open_min(self, new_value: float):
        self.jaw_open_min = new_value

    def set_jaw_open_max(self, new_value: float):
        self.jaw_open_max = new_value

    def set_mouth_frown_max(self, new_value: float):
        self.mouth_frown_max = new_value

    def set_mouth_funnel_min(self, new_value: float):
        self.mouth_funnel_min = new_value

    def set_mouth_funnel_max(self, new_value: float):
        self.mouth_funnel_min = new_value


class MediaPoseFacePoseConverter00(MediaPipeFacePoseConverter):
    def __init__(self, args: Optional[MediaPipeFacePoseConverter00Args] = None):
        super().__init__()
        if args is None:
            args = MediaPipeFacePoseConverter00Args()
        self.args = args
        pose_parameters = get_pose_parameters()
        self.pose_size = 45

        self.eyebrow_troubled_left_index = pose_parameters.get_parameter_index("eyebrow_troubled_left")
        self.eyebrow_troubled_right_index = pose_parameters.get_parameter_index("eyebrow_troubled_right")
        self.eyebrow_angry_left_index = pose_parameters.get_parameter_index("eyebrow_angry_left")
        self.eyebrow_angry_right_index = pose_parameters.get_parameter_index("eyebrow_angry_right")
        self.eyebrow_happy_left_index = pose_parameters.get_parameter_index("eyebrow_happy_left")
        self.eyebrow_happy_right_index = pose_parameters.get_parameter_index("eyebrow_happy_right")
        self.eyebrow_raised_left_index = pose_parameters.get_parameter_index("eyebrow_raised_left")
        self.eyebrow_raised_right_index = pose_parameters.get_parameter_index("eyebrow_raised_right")
        self.eyebrow_lowered_left_index = pose_parameters.get_parameter_index("eyebrow_lowered_left")
        self.eyebrow_lowered_right_index = pose_parameters.get_parameter_index("eyebrow_lowered_right")
        self.eyebrow_serious_left_index = pose_parameters.get_parameter_index("eyebrow_serious_left")
        self.eyebrow_serious_right_index = pose_parameters.get_parameter_index("eyebrow_serious_right")

        self.eye_surprised_left_index = pose_parameters.get_parameter_index("eye_surprised_left")
        self.eye_surprised_right_index = pose_parameters.get_parameter_index("eye_surprised_right")
        self.eye_wink_left_index = pose_parameters.get_parameter_index("eye_wink_left")
        self.eye_wink_right_index = pose_parameters.get_parameter_index("eye_wink_right")
        self.eye_happy_wink_left_index = pose_parameters.get_parameter_index("eye_happy_wink_left")
        self.eye_happy_wink_right_index = pose_parameters.get_parameter_index("eye_happy_wink_right")
        self.eye_relaxed_left_index = pose_parameters.get_parameter_index("eye_relaxed_left")
        self.eye_relaxed_right_index = pose_parameters.get_parameter_index("eye_relaxed_right")
        self.eye_raised_lower_eyelid_left_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_left")
        self.eye_raised_lower_eyelid_right_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_right")

        self.iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
        self.iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")

        self.iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
        self.iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")

        self.head_x_index = pose_parameters.get_parameter_index("head_x")
        self.head_y_index = pose_parameters.get_parameter_index("head_y")
        self.neck_z_index = pose_parameters.get_parameter_index("neck_z")

        self.mouth_aaa_index = pose_parameters.get_parameter_index("mouth_aaa")
        self.mouth_iii_index = pose_parameters.get_parameter_index("mouth_iii")
        self.mouth_uuu_index = pose_parameters.get_parameter_index("mouth_uuu")
        self.mouth_eee_index = pose_parameters.get_parameter_index("mouth_eee")
        self.mouth_ooo_index = pose_parameters.get_parameter_index("mouth_ooo")

        self.mouth_lowered_corner_left_index = pose_parameters.get_parameter_index("mouth_lowered_corner_left")
        self.mouth_lowered_corner_right_index = pose_parameters.get_parameter_index("mouth_lowered_corner_right")
        self.mouth_raised_corner_left_index = pose_parameters.get_parameter_index("mouth_raised_corner_left")
        self.mouth_raised_corner_right_index = pose_parameters.get_parameter_index("mouth_raised_corner_right")

        self.body_y_index = pose_parameters.get_parameter_index("body_y")
        self.body_z_index = pose_parameters.get_parameter_index("body_z")
        self.breathing_index = pose_parameters.get_parameter_index("breathing")

        self.breathing_start_time = time.time()

        self.panel = None
        self.current_pose_supplier = None

    def init_pose_converter_panel(
            self,
            parent,
            current_pose_supplier: Callable[[], Optional[MediaPipeFacePose]]):
        self.panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        self.panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.panel_sizer)
        self.panel.SetAutoLayout(1)
        parent.GetSizer().Add(self.panel, 0, wx.EXPAND)

        self.current_pose_supplier = current_pose_supplier

        if True:
            eyebrow_down_mode_text = wx.StaticText(self.panel, label=" --- Eyebrow Down Mode --- ",
                                                   style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(eyebrow_down_mode_text, 0, wx.EXPAND)

            self.eyebrow_down_mode_choice = wx.Choice(
                self.panel,
                choices=[
                    "ANGRY",
                    "TROUBLED",
                    "SERIOUS",
                    "LOWERED",
                ])
            self.eyebrow_down_mode_choice.SetSelection(0)
            self.panel_sizer.Add(self.eyebrow_down_mode_choice, 0, wx.EXPAND)
            self.eyebrow_down_mode_choice.Bind(wx.EVT_CHOICE, self.change_eyebrow_down_mode)

        if True:
            separator = wx.StaticLine(self.panel, -1, size=(256, 5))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

            wink_mode_text = wx.StaticText(self.panel, label=" --- Wink Mode --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(wink_mode_text, 0, wx.EXPAND)

            self.wink_mode_choice = wx.Choice(
                self.panel,
                choices=[
                    "NORMAL",
                    "RELAXED",
                ])
            self.wink_mode_choice.SetSelection(0)
            self.panel_sizer.Add(self.wink_mode_choice, 0, wx.EXPAND)
            self.wink_mode_choice.Bind(wx.EVT_CHOICE, self.change_wink_mode)

        if True:
            separator = wx.StaticLine(self.panel, -1, size=(256, 5))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

            iris_size_text = wx.StaticText(self.panel, label=" --- Iris Size --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(iris_size_text, 0, wx.EXPAND)

            self.iris_left_slider = wx.Slider(self.panel, minValue=0, maxValue=1000, value=0, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.iris_left_slider, 0, wx.EXPAND)
            self.iris_left_slider.Bind(wx.EVT_SLIDER, self.change_iris_size)

            self.iris_right_slider = wx.Slider(self.panel, minValue=0, maxValue=1000, value=0, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.iris_right_slider, 0, wx.EXPAND)
            self.iris_right_slider.Bind(wx.EVT_SLIDER, self.change_iris_size)
            self.iris_right_slider.Enable(False)

            self.link_left_right_irises = wx.CheckBox(
                self.panel, label="Use same value for both sides")
            self.link_left_right_irises.SetValue(True)
            self.panel_sizer.Add(self.link_left_right_irises, wx.SizerFlags().CenterHorizontal().Border())
            self.link_left_right_irises.Bind(wx.EVT_CHECKBOX, self.link_left_right_irises_clicked)

        if True:
            separator = wx.StaticLine(self.panel, -1, size=(256, 5))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

            breathing_frequency_text = wx.StaticText(
                self.panel, label=" --- Breathing --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(breathing_frequency_text, 0, wx.EXPAND)

            self.restart_breathing_cycle_button = wx.Button(self.panel, label="Restart Breathing Cycle")
            self.restart_breathing_cycle_button.Bind(wx.EVT_BUTTON, self.restart_breathing_cycle_clicked)
            self.panel_sizer.Add(self.restart_breathing_cycle_button, 0, wx.EXPAND)

            self.breathing_frequency_slider = wx.Slider(
                self.panel, minValue=0, maxValue=60, value=20, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.breathing_frequency_slider, 0, wx.EXPAND)

            self.breathing_gauge = wx.Gauge(self.panel, style=wx.GA_HORIZONTAL, range=1000)
            self.panel_sizer.Add(self.breathing_gauge, 0, wx.EXPAND)

        if True:
            separator = wx.StaticLine(self.panel, -1, size=(256, 5))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

            face_orientation_text = wx.StaticText(
                self.panel, label="--- Face Orientation ---", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(face_orientation_text, 0, wx.EXPAND)

            self.calibrate_face_orientation_button = wx.Button(self.panel, label="Calibrate (I'm looking forward)")
            self.calibrate_face_orientation_button.Bind(wx.EVT_BUTTON, self.calibrate_face_orientation_clicked)
            self.panel_sizer.Add(self.calibrate_face_orientation_button, 0, wx.EXPAND)

        if True:
            separator = wx.StaticLine(self.panel, -1, size=(256, 5))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

            convertion_parameters_text = wx.StaticText(
                self.panel, label="--- Conversion Parameters ---", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(convertion_parameters_text, 0, wx.EXPAND)

            conversion_param_panel = wx.Panel(self.panel)
            self.panel_sizer.Add(conversion_param_panel, 0, wx.EXPAND)
            conversion_panel_sizer = wx.FlexGridSizer(cols=2)
            conversion_panel_sizer.AddGrowableCol(1)
            conversion_param_panel.SetSizer(conversion_panel_sizer)
            conversion_param_panel.SetAutoLayout(1)

            self.smile_thresold_min_spin = self.create_spin_control(
                conversion_param_panel,
                "Smile Threshold Min:", self.args.smile_threshold_min, self.args.set_smile_threshold_min)
            self.smile_thresold_max_spin = self.create_spin_control(
                conversion_param_panel,
                "Smile Threshold Max:", self.args.smile_threshold_max, self.args.set_smile_threshold_max)
            self.eye_surprised_max_spin = self.create_spin_control(
                conversion_param_panel,
                "Eye Surprised Max:", self.args.eye_surprised_max, self.args.set_eye_surprised_max)
            self.eye_blink_max_spin = self.create_spin_control(
                conversion_param_panel,
                "Eye Blink Max:", self.args.eye_blink_max, self.args.set_eye_blink_max)
            self.eyebrow_down_max_spin = self.create_spin_control(
                conversion_param_panel,
                "Eyebrow Down Max:", self.args.eyebrow_down_max, self.args.set_eyebrow_down_max)
            self.cheek_squint_min_spin = self.create_spin_control(
                conversion_param_panel,
                "Cheek Squint Min:", self.args.cheek_squint_min, self.args.set_cheek_squint_min)
            self.cheek_squint_max_spin = self.create_spin_control(
                conversion_param_panel,
                "Cheek Squint Max:", self.args.cheek_squint_max, self.args.set_cheek_squint_max)
            self.jaw_open_min_spin = self.create_spin_control(
                conversion_param_panel,
                "Jaw Open Min:", self.args.jaw_open_min, self.args.set_jaw_open_min)
            self.jaw_open_max_spin = self.create_spin_control(
                conversion_param_panel,
                "Jaw Open Max:", self.args.jaw_open_max, self.args.set_jaw_open_max)
            self.mouth_frown_max_spin = self.create_spin_control(
                conversion_param_panel,
                "Mouth Frown Max:", self.args.mouth_frown_max, self.args.set_mouth_frown_max)
            self.mouth_funnel_min_spin = self.create_spin_control(
                conversion_param_panel,
                "Mouth Funnel Min:", self.args.mouth_funnel_min, self.args.set_mouth_funnel_min)
            self.mouth_funnel_max_spin = self.create_spin_control(
                conversion_param_panel,
                "Mouth Funnel Max:", self.args.mouth_funnel_max, self.args.set_mouth_funnel_max)

        self.panel_sizer.Fit(self.panel)

    def create_spin_control(self, parent, label: str, initial_value: float, set_func: Callable[[float], None]):
        sizer = parent.GetSizer()

        text = wx.StaticText(parent, label=label)
        sizer.Add(text, wx.SizerFlags().Right().Border(wx.ALL, 2))

        spin_ctrl = wx.SpinCtrlDouble(
            parent,
            wx.ID_ANY,
            min=0.0,
            max=1.0,
            initial=initial_value,
            inc=0.01)
        sizer.Add(spin_ctrl, wx.SizerFlags().Border(wx.ALL, 2).Expand())

        def handler(event: wx.Event):
            new_value = spin_ctrl.GetValue()
            set_func(new_value)

        spin_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, handler)

        return spin_ctrl

    def extract_euler_angles(self, mediapipe_face_pose: MediaPipeFacePose):
        M = mediapipe_face_pose.xform_matrix[0:3, 0:3]
        rot = Rotation.from_matrix(M)
        return rot.as_euler('xyz', degrees=False)

    def calibrate_face_orientation_clicked(self, event: wx.Event):
        if self.current_pose_supplier is None:
            return

        mediapipe_face_pose = self.current_pose_supplier()
        if mediapipe_face_pose is None:
            return

        euler_angles = self.extract_euler_angles(mediapipe_face_pose)
        self.args.head_x_offset = euler_angles[0]
        self.args.head_y_offset = euler_angles[1]
        self.args.head_z_offset = euler_angles[2]

    def restart_breathing_cycle_clicked(self, event: wx.Event):
        self.breathing_start_time = time.time()

    def change_eyebrow_down_mode(self, event: wx.Event):
        selected_index = self.eyebrow_down_mode_choice.GetSelection()
        if selected_index == 0:
            self.args.eyebrow_down_mode = EyebrowDownMode.ANGRY
        elif selected_index == 1:
            self.args.eyebrow_down_mode = EyebrowDownMode.TROUBLED
        elif selected_index == 2:
            self.args.eyebrow_down_mode = EyebrowDownMode.SERIOUS
        else:
            self.args.eyebrow_down_mode = EyebrowDownMode.LOWERED

    def change_wink_mode(self, event: wx.Event):
        selected_index = self.wink_mode_choice.GetSelection()
        if selected_index == 0:
            self.args.wink_mode = WinkMode.NORMAL
        else:
            self.args.wink_mode = WinkMode.RELAXED

    def change_iris_size(self, event: wx.Event):
        if self.link_left_right_irises.GetValue():
            left_value = self.iris_left_slider.GetValue()
            right_value = self.iris_right_slider.GetValue()
            if left_value != right_value:
                self.iris_right_slider.SetValue(left_value)
            self.args.iris_small_left = left_value / 1000.0
            self.args.iris_small_right = left_value / 1000.0
        else:
            self.args.iris_small_left = self.iris_left_slider.GetValue() / 1000.0
            self.args.iris_small_right = self.iris_right_slider.GetValue() / 1000.0

    def link_left_right_irises_clicked(self, event: wx.Event):
        if self.link_left_right_irises.GetValue():
            self.iris_right_slider.Enable(False)
        else:
            self.iris_right_slider.Enable(True)
        self.change_iris_size(event)

    def decompose_head_body_param(self, param, threshold=2.0 / 3):
        if abs(param) < threshold:
            return (param, 0.0)
        else:
            if param < 0:
                sign = -1.0
            else:
                sign = 1.0
            return (threshold * sign, (abs(param) - threshold) * sign)

    def convert(self, mediapipe_face_pose: MediaPipeFacePose) -> List[float]:
        pose = [0.0 for i in range(self.pose_size)]

        blendshape_params = mediapipe_face_pose.blendshape_params

        smile_value = \
            (blendshape_params[MOUTH_SMILE_LEFT] + blendshape_params[MOUTH_SMILE_RIGHT]) / 2.0 \
            + blendshape_params[MOUTH_SHRUG_UPPER]
        if self.args.smile_threshold_min >= self.args.smile_threshold_max:
            smile_degree = 0.0
        else:
            if smile_value < self.args.smile_threshold_min:
                smile_degree = 0.0
            elif smile_value > self.args.smile_threshold_max:
                smile_degree = 1.0
            else:
                smile_degree = (smile_value - self.args.smile_threshold_min) / (
                        self.args.smile_threshold_max - self.args.smile_threshold_min)

        # Eyebrow
        if True:
            brow_inner_up = blendshape_params[BROW_INNER_UP]
            brow_outer_up_right = blendshape_params[BROW_OUTER_UP_RIGHT]
            brow_outer_up_left = blendshape_params[BROW_OUTER_UP_LEFT]

            brow_up_left = clamp(brow_inner_up + brow_outer_up_left, 0.0, 1.0)
            brow_up_right = clamp(brow_inner_up + brow_outer_up_right, 0.0, 1.0)
            pose[self.eyebrow_raised_left_index] = brow_up_left
            pose[self.eyebrow_raised_right_index] = brow_up_right

            if self.args.eyebrow_down_max <= 0.0:
                brow_down_left = 0.0
                brow_down_right = 0.0
            else:
                brow_down_left = (1.0 - smile_degree) \
                                 * clamp(blendshape_params[BROW_DOWN_LEFT] / self.args.eyebrow_down_max, 0.0, 1.0)
                brow_down_right = (1.0 - smile_degree) \
                                  * clamp(blendshape_params[BROW_DOWN_RIGHT] / self.args.eyebrow_down_max, 0.0, 1.0)

            if self.args.eyebrow_down_mode == EyebrowDownMode.TROUBLED:
                pose[self.eyebrow_troubled_left_index] = brow_down_left
                pose[self.eyebrow_troubled_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.ANGRY:
                pose[self.eyebrow_angry_left_index] = brow_down_left
                pose[self.eyebrow_angry_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.LOWERED:
                pose[self.eyebrow_lowered_left_index] = brow_down_left
                pose[self.eyebrow_lowered_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.SERIOUS:
                pose[self.eyebrow_serious_left_index] = brow_down_left
                pose[self.eyebrow_serious_right_index] = brow_down_right

            brow_happy_value = clamp(smile_value, 0.0, 1.0) * smile_degree
            pose[self.eyebrow_happy_left_index] = brow_happy_value
            pose[self.eyebrow_happy_right_index] = brow_happy_value

        # Eye
        if True:
            # Surprised
            if self.args.eye_surprised_max <= 0.0:
                pose[self.eye_surprised_left_index] = 0.0
                pose[self.eye_surprised_right_index] = 0.0
            else:
                pose[self.eye_surprised_left_index] = clamp(
                    blendshape_params[EYE_WIDE_LEFT] / self.args.eye_surprised_max, 0.0, 1.0)
                pose[self.eye_surprised_right_index] = clamp(
                    blendshape_params[EYE_WIDE_RIGHT] / self.args.eye_surprised_max, 0.0, 1.0)

            # Wink
            if self.args.wink_mode == WinkMode.NORMAL:
                wink_left_index = self.eye_wink_left_index
                wink_right_index = self.eye_wink_right_index
            else:
                wink_left_index = self.eye_relaxed_left_index
                wink_right_index = self.eye_relaxed_right_index
            if self.args.eye_blink_max <= 0:
                pose[wink_left_index] = 0.0
                pose[wink_right_index] = 0.0
                pose[self.eye_happy_wink_left_index] = 0.0
                pose[self.eye_happy_wink_right_index] = 0.0
            else:
                pose[wink_left_index] = (1.0 - smile_degree) * clamp(
                    blendshape_params[EYE_BLINK_LEFT] / self.args.eye_blink_max, 0.0, 1.0)
                pose[wink_right_index] = (1.0 - smile_degree) * clamp(
                    blendshape_params[EYE_BLINK_RIGHT] / self.args.eye_blink_max, 0.0, 1.0)
                pose[self.eye_happy_wink_left_index] = smile_degree * clamp(
                    blendshape_params[EYE_BLINK_LEFT] / self.args.eye_blink_max, 0.0, 1.0)
                pose[self.eye_happy_wink_right_index] = smile_degree * clamp(
                    blendshape_params[EYE_BLINK_RIGHT] / self.args.eye_blink_max, 0.0, 1.0)

            # Lower eyelid
            cheek_squint_denom = self.args.cheek_squint_max - self.args.cheek_squint_min
            if cheek_squint_denom <= 0.0:
                pose[self.eye_raised_lower_eyelid_left_index] = 0.0
                pose[self.eye_raised_lower_eyelid_right_index] = 0.0
            else:
                pose[self.eye_raised_lower_eyelid_left_index] = \
                    clamp(
                        (blendshape_params[CHEEK_SQUINT_LEFT] - self.args.cheek_squint_min) / cheek_squint_denom,
                        0.0, 1.0)
                pose[self.eye_raised_lower_eyelid_right_index] = \
                    clamp(
                        (blendshape_params[CHEEK_SQUINT_RIGHT] - self.args.cheek_squint_min) / cheek_squint_denom,
                        0.0, 1.0)

        # Iris rotation
        if True:
            eye_rotation_y = (blendshape_params[EYE_LOOK_IN_LEFT]
                              - blendshape_params[EYE_LOOK_OUT_LEFT]
                              - blendshape_params[EYE_LOOK_IN_RIGHT]
                              + blendshape_params[EYE_LOOK_OUT_RIGHT]) / 2.0 * self.args.eye_rotation_factor
            pose[self.iris_rotation_y_index] = clamp(eye_rotation_y, -1.0, 1.0)

            eye_rotation_x = (blendshape_params[EYE_LOOK_UP_LEFT]
                              + blendshape_params[EYE_LOOK_UP_RIGHT]
                              - blendshape_params[EYE_LOOK_DOWN_LEFT]
                              - blendshape_params[EYE_LOOK_DOWN_RIGHT]) / 2.0 * self.args.eye_rotation_factor
            pose[self.iris_rotation_x_index] = clamp(eye_rotation_x, -1.0, 1.0)

        # Iris size
        if True:
            pose[self.iris_small_left_index] = self.args.iris_small_left
            pose[self.iris_small_right_index] = self.args.iris_small_right

        # Head rotation
        if True:
            euler_angles = self.extract_euler_angles(mediapipe_face_pose)
            euler_angles[0] -= self.args.head_x_offset
            euler_angles[1] -= self.args.head_y_offset
            euler_angles[2] -= self.args.head_z_offset

            x_param = clamp(-euler_angles[0] * 180.0 / math.pi, -15.0, 15.0) / 15.0
            pose[self.head_x_index] = x_param

            y_param = clamp(-euler_angles[1] * 180.0 / math.pi, -10.0, 10.0) / 10.0
            pose[self.head_y_index] = y_param
            pose[self.body_y_index] = y_param

            z_param = clamp(euler_angles[2] * 180.0 / math.pi, -15.0, 15.0) / 15.0
            pose[self.neck_z_index] = z_param
            pose[self.body_z_index] = z_param

        # Mouth
        if True:
            jaw_open_denom = self.args.jaw_open_max - self.args.jaw_open_min
            if jaw_open_denom <= 0:
                mouth_open = 0.0
            else:
                mouth_open = clamp((blendshape_params[JAW_OPEN] - self.args.jaw_open_min) / jaw_open_denom, 0.0, 1.0)
            pose[self.mouth_aaa_index] = mouth_open
            pose[self.mouth_raised_corner_left_index] = clamp(smile_value, 0.0, 1.0)
            pose[self.mouth_raised_corner_right_index] = clamp(smile_value, 0.0, 1.0)

            is_mouth_open = mouth_open > 0.0
            if not is_mouth_open:
                if self.args.mouth_frown_max <= 0:
                    mouth_frown_value = 0.0
                else:
                    mouth_frown_value = clamp(
                        (blendshape_params[MOUTH_FROWN_LEFT] + blendshape_params[
                            MOUTH_FROWN_RIGHT]) / self.args.mouth_frown_max, 0.0, 1.0)
                pose[self.mouth_lowered_corner_left_index] = mouth_frown_value
                pose[self.mouth_lowered_corner_right_index] = mouth_frown_value
            else:
                mouth_lower_down = clamp(
                    blendshape_params[MOUTH_LOWER_DOWN_LEFT] + blendshape_params[MOUTH_LOWER_DOWN_RIGHT], 0.0, 1.0)
                mouth_funnel = blendshape_params[MOUTH_FUNNEL]
                mouth_pucker = blendshape_params[MOUTH_PUCKER]

                mouth_point = [mouth_open, mouth_lower_down, mouth_funnel, mouth_pucker]

                aaa_point = [1.0, 1.0, 0.0, 0.0]
                iii_point = [0.0, 1.0, 0.0, 0.0]
                uuu_point = [0.5, 0.3, 0.25, 0.75]
                ooo_point = [1.0, 0.5, 0.5, 0.4]

                decomp = numpy.array([0, 0, 0, 0])
                M = numpy.array([
                    aaa_point,
                    iii_point,
                    uuu_point,
                    ooo_point
                ])

                def loss(decomp):
                    return numpy.linalg.norm(numpy.matmul(decomp, M) - mouth_point) \
                        + 0.01 * numpy.linalg.norm(decomp, ord=1)

                opt_result = scipy.optimize.minimize(
                    loss, decomp, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
                decomp = opt_result["x"]
                restricted_decomp = [decomp.item(0), decomp.item(1), decomp.item(2), decomp.item(3)]
                pose[self.mouth_aaa_index] = restricted_decomp[0]
                pose[self.mouth_iii_index] = restricted_decomp[1]
                mouth_funnel_denom = self.args.mouth_funnel_max - self.args.mouth_funnel_min
                if mouth_funnel_denom <= 0:
                    ooo_alpha = 0.0
                    uo_value = 0.0
                else:
                    ooo_alpha = clamp((mouth_funnel - self.args.mouth_funnel_min) / mouth_funnel_denom, 0.0, 1.0)
                    uo_value = clamp(restricted_decomp[2] + restricted_decomp[3], 0.0, 1.0)
                pose[self.mouth_uuu_index] = uo_value * (1.0 - ooo_alpha)
                pose[self.mouth_ooo_index] = uo_value * ooo_alpha

        if self.panel is not None:
            frequency = self.breathing_frequency_slider.GetValue()
            if frequency == 0:
                value = 0.0
                pose[self.breathing_index] = value
                self.breathing_start_time = time.time()
            else:
                period = 60.0 / frequency
                now = time.time()
                diff = now - self.breathing_start_time
                frac = (diff % period) / period
                value = (-math.cos(2 * math.pi * frac) + 1.0) / 2.0
                pose[self.breathing_index] = value
            self.breathing_gauge.SetValue(int(1000 * value))

        return pose