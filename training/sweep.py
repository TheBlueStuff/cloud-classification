import subprocess
import sys
import os
import argparse

REQUIREMENTS_PATH = '/opt/ml/processing/input/requirements'
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",
                        os.path.join(REQUIREMENTS_PATH, "requirements_aws.txt")])

# clone repository
subprocess.run(["git", "clone", "https://github.com/TheBlueStuff/cloud-classification.git"])

# get wandb sweep id
parser = argparse.ArgumentParser()
parser.add_argument('--sweep_id', type=str, default='')
args = parser.parse_args()
sweep_id = args.sweep_id

# run wandb sweep
os.chdir("cloud-classification/training")
sweep_str = "mario-andonaire/cloud-classification/{}".format(sweep_id)
subprocess.run(["wandb", "agent", sweep_str])
