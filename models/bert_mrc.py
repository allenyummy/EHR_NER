# encoding=utf-8
# Author: Allen.Chiang
# Description: bert model for machine reading comprehension

import logging
from torch.nn import Linear, CrossEntropyLoss
from transformers import BertConfig, BertPreTrainedModel, BertModel
from utils.modeling_output import QuestionAnsweringModelOutput

logger = logging.getLogger(__name__)


class BertMRCModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.start_outputs = Linear(config.hidden_size, 2)
        self.end_outoputs = Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        start_logits = self.start_outputs(sequence_output)
        end_logits = self.end_outoputs(sequence_output)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
