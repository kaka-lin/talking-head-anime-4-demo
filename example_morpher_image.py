from PIL import Image
import matplotlib.pyplot as plt
import torch

from tha4.charmodel.character_model import CharacterModel
from tha4.poser.modes.mode_14 import get_pose_parameters
from tha4.nn.siren.face_morpher.siren_face_morpher_00 import SirenFaceMorpher00Args, SirenFaceMorpher00
from tha4.nn.siren.morpher.siren_morpher_03 import SirenMorpher03, SirenMorpher03Args, SirenMorpherLevelArgs
from tha4.nn.siren.vanilla.siren import SirenArgs, Siren
from tha4.nn.siren.vanilla.siren import SirenArgs, Siren
from tha4.image_util import convert_output_image_from_torch_to_numpy
from tha4.shion.base.image_util import extract_pytorch_image_from_PIL_image


if __name__ == "__main__":
    face_file_name = "data/character_models/lambda_01/face_morpher.pt"
    body_file_name = "data/character_models/lambda_01/body_morpher.pt"

    # Load the Original image
    original_image = Image.open("data/character_models/lambda_01/character.png")
    torch_source_image = extract_pytorch_image_from_PIL_image(original_image)
    torch_source_image = torch_source_image.unsqueeze(0)

    face_morpher_module = SirenFaceMorpher00(
        SirenFaceMorpher00Args(
            image_size=128,
            image_channels=4,
            pose_size=39,
            siren_args=SirenArgs(
                in_channels=39 + 2,
                out_channels=4,
                intermediate_channels=128,
                num_sine_layers=8)))
    
    with open(face_file_name, "rb") as f:
        state_dict = torch.load(f, map_location=lambda storage, loc: storage)
    face_morpher_module.load_state_dict(state_dict)
    
    current_pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9605, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7065, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.373, 0.6890000000000001, 0.879, 0.42500000000000004, 0.0]
    pose = torch.tensor(current_pose)
    pose = pose.unsqueeze(0)
    facee_pose = pose[:, :39]

    with torch.no_grad():
        face_morphed_image = face_morpher_module(facee_pose)
    face_numpy_image = convert_output_image_from_torch_to_numpy(face_morphed_image[0])
    
    center_x = 256
    center_y = 128 + 16
    image = torch_source_image.clone()
    image[:, :, center_y - 64:center_y + 64, center_x - 64:center_x + 64] = face_morphed_image

    body_morpher_module = SirenMorpher03(
        SirenMorpher03Args(
            image_size=512,
            image_channels=4,
            pose_size=45,
            level_args=[
                SirenMorpherLevelArgs(
                    image_size=128,
                    intermediate_channels=360,
                    num_sine_layers=3),
                SirenMorpherLevelArgs(
                    image_size=256,
                    intermediate_channels=180,
                    num_sine_layers=3),
                SirenMorpherLevelArgs(
                    image_size=512,
                    intermediate_channels=90,
                    num_sine_layers=3),
            ]))

    with open(body_file_name, "rb") as f:
        state_dict = torch.load(f, map_location=lambda storage, loc: storage)
    body_morpher_module.load_state_dict(state_dict)
        
    with torch.no_grad():
        body_morpher_image = body_morpher_module.forward(image, pose)
    body_numpy_image = convert_output_image_from_torch_to_numpy(body_morpher_image[0][0])

    # Visualize the results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
    ax1.imshow(original_image)
    ax1.axis("off")
    ax2.imshow(face_numpy_image)
    ax2.axis("off")
    ax3.imshow(body_numpy_image)
    ax3.axis("off")
    fig.tight_layout()
    plt.show()
