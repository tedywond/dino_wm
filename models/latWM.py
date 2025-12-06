import sys
import os
sys.path.append('/home/ripl-pc/Desktop/latent_action/latAct')

from libs.uvit import UViT as LatActDiffusionWM
from spatio_temporal_encoder import get_spatio_temporal_encoder as get_st_encoder
import sde
from ds_util import invTrans
from torchvision import transforms

transform=transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

def get_score_model(nnet, pred=True):
    score_model = sde.ScoreModel(nnet, pred='x0_pred_t', sde=sde.VPSDE())
    return score_model