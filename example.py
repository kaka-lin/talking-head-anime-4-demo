import PIL.Image
import matplotlib.pyplot as plt
import torch
import numpy as np

from tha4.charmodel.character_model import CharacterModel
from tha4.image_util import resize_PIL_image, convert_output_image_from_torch_to_numpy


if __name__ == "__main__":
    # for CUDA or MPS (Metal GPU)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        # model not supported on MPS,
        # so we use CPU
        device = "cpu" 

    # Model configuration
    NUM_PARAMETERS = 45
    character_model_file_name = "data/character_models/lambda_01/character_model.yaml"

    # Load the character model that you want
    character_model = CharacterModel.load(character_model_file_name)
    # Get the face and body model
    poser = character_model.get_poser(device)

    # Load the Original image
    original_image = resize_PIL_image(
        PIL.Image.open(character_model.character_image_file_name), (512, 512))

    # Get input image
    torch_source_image = character_model.get_character_image(device)

    # Get the input pose vector
    # current_pose = [0.0 for i in range(NUM_PARAMETERS)]
    current_pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9605, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7065, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.373, 0.6890000000000001, 0.879, 0.42500000000000004, 0.0]
    pose = torch.tensor(current_pose)

    # Inference
    output_image = poser.pose(torch_source_image, pose)[0].float()
    numpy_image = convert_output_image_from_torch_to_numpy(output_image)
    
    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(original_image)
    ax1.axis("off")
    ax2.imshow(numpy_image)
    ax2.axis("off")
    fig.tight_layout()
    plt.show()