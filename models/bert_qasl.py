# encoding=utf-8
# Author: Allen.Chiang
# Description: bert model for QA sequence labeling

import logging
from typing import List, Optional
import torch
from torch.nn import Linear, Dropout, CrossEntropyLoss
from transformers import BertConfig, BertPreTrainedModel, BertModel
from utils.modeling_output import TokenClassificationModelOutput

logger = logging.getLogger(__name__)


class BertQASLModel(BertPreTrainedModel):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.return_dict = (
            config.return_dict if hasattr(config, "return_dict") else False
        )
        self.class_weights = (
            torch.FloatTensor(class_weights) if class_weights else None
        )
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=self.return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.cuda())
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.return_dict:
            return TokenClassificationModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
