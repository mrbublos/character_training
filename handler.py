import os
import runpod
import subprocess
import yaml
from runpod.serverless.modules.rp_logger import RunPodLogger
from runpod.serverless.utils.rp_validator import validate

from preprocess_images import start
from schema import INPUT_SCHEMA

logger = RunPodLogger()

BASE_DIR = os.getenv("BASE_DIR")

RAW_IMAGES_DIR = "user_data"
PROCESSED_IMAGES_DIR = "user_datasets"
TRAINED_MODELS_DIR = "user_models"
DEFAULT_CONFIG = './config/train_config_1h100.yaml'

def create_config(user_id, steps):
    logger.info(f"Updating config for {user_id}")
    with open(DEFAULT_CONFIG, 'r') as file:
        config = yaml.safe_load(file)

    old_input_folder = config['config']['process'][0]['datasets'][0]['folder_path']
    input_folder = f"{BASE_DIR}/{PROCESSED_IMAGES_DIR}/{user_id}"
    config['config']['process'][0]['datasets'][0]['folder_path'] = input_folder
    logger.debug(f"Updated config input folder to {input_folder} from {old_input_folder}")

    old_output_folder = config['config']['name']
    output_folder = user_id
    config['config']['name'] = output_folder
    logger.debug(f"Updated config output folder to {output_folder} from {old_output_folder}")

    old_steps = config['config']['process'][0]['train']['steps']
    new_steps = steps or old_steps
    config['config']['process'][0]['train']['steps'] = new_steps
    logger.debug(f"Updated config steps from {old_steps} to {new_steps}")

    old_training_folder = config['config']['process'][0]['training_folder']
    new_training_folder = f"{BASE_DIR}/{TRAINED_MODELS_DIR}"
    config['config']['process'][0]['training_folder'] = new_training_folder
    logger.debug(f"Updated config training_folder from {old_training_folder} to {new_training_folder}")

    config_name = f"{BASE_DIR}/{RAW_IMAGES_DIR}/{user_id}/config.yaml"
    with open(config_name, 'w') as file:
        yaml.dump(config, file)

    return config_name

def handler(event):
    job_id = event['id']    

    validated_input = validate(event['input'], INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {
            'error': validated_input['errors']
        }

    input = validated_input["validated_input"]
    user_id = input['user_id']

    try:
        user_photos_raw = f"{BASE_DIR}/{RAW_IMAGES_DIR}/{user_id}"
        user_photos_processed = f"{BASE_DIR}/{PROCESSED_IMAGES_DIR}/{user_id}"

        os.makedirs(os.path.dirname(user_photos_raw + "/"), exist_ok=True)
        os.makedirs(os.path.dirname(user_photos_processed + "/"), exist_ok=True)

        config_name = create_config(user_id=user_id, steps=input['steps'])
        script_path = f"./start_training.sh"

        logger.debug(f"Starting training for {user_id}")

        logger.info(f"Preprocessing images {user_id}")
        start(input_dir=user_photos_raw,
            output_dir=user_photos_processed)
    
        logger.info(f"Learning model for {user_id}")
        result = subprocess.run([script_path, config_name], capture_output=False, text=True)
        logger.debug(f"Completed training for {user_id}")
        # Check the script's exit code
        if result.returncode == 0:
            return {
                'job_id': job_id,
                'user_id': user_id,
                'success': True,
            }
        else:
            return {
                'job_id': job_id,
                'user_id': user_id,
                'success': False,
            }


    except Exception as e:
        logger.error(f"Error training model file for {user_id}: {e}", job_id)
        return {
            "success": False,
        }

if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})