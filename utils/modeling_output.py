# encoding=utf-8
# Author: Allen.Chiang
# Description: model output format

import logging
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenClassificationModelOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class QuestionAnsweringModelOutput:
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None