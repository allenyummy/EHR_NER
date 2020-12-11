# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: bert model for QA sequence labeling

import logging
from typing import List, Optional
import torch
from torch.nn import Linear, LSTM, Dropout, CrossEntropyLoss
from transformers import BertConfig, BertPreTrainedModel, BertModel
from utils.modeling_output import TokenClassificationModelOutput
from models.crf_layer import CRF

logger = logging.getLogger(__name__)


class BertBiLSTMCRFQASLModel(BertPreTrainedModel):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.return_dict = (
            config.return_dict if hasattr(config, "return_dict") else False
        )
        self.class_weights = torch.FloatTensor(class_weights) if class_weights else None
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.lstm = LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            bidirectional=True,
            # dropout=0.5   ## dropout only works when num_layer is greater than 1.
        )
        ## So I explicitly add dropout for lstm outside.
        self.dropout_lstm = Dropout(0.5)
        self.classifier = Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

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
        lstm_output = self.lstm(sequence_output)
        lstm_output = lstm_output[0]
        lstm_output = self.dropout_lstm(lstm_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            ## Find the tokens that labelid is -100. Actually, they are [CLS], [SEP], and [PAD].
            active_idx = labels != CrossEntropyLoss().ignore_index
            ## Replace them into the labelid of "O"
            active_labels = torch.where(
                active_idx,
                labels,
                torch.tensor(self.config.label2id["O"]).type_as(labels),
            )
            ## Log Likelihood
            ## [CLS] and [SEP] do contribute to loss.
            ## [PAD] do not contribute to loss due to masking.
            loss = self.crf(
                emissions=logits,     ## [TBD] need to be weighted.
                tags=active_labels,
                mask=attention_mask.type(torch.uint8),
            )
            ## Negative Log Likelihood
            loss = -1 * loss

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
