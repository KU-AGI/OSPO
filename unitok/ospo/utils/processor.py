
import torch
import pyrootutils
import numpy as np
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def get_conversation(prompt: str, cfg_prob: float=0.1):
    
    assert cfg_prob >= 0 and cfg_prob <= 1, "cfg_prob must be between 0 and 1"

    if np.random.rand() < cfg_prob:
        conversation = prompt + ' Generate an image based on this description.'
    else:
        conversation = '<unconditional>'

    return conversation

