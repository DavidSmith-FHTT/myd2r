import torch
from torch import nn
from .modeling_unimo import UnimoModel
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class UnimoModelF(nn.Module):
    def __init__(self, args, vision_config, text_config):
        super(UnimoModelF, self).__init__()
        self.args = args
        self.vision_config = vision_config
        self.text_config = text_config
        self.model = UnimoModel(args, vision_config, text_config)
        self.fc = nn.Linear(self.text_config.hidden_size, 3)
        self.CE_Loss = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels, images):
        output, js_loss = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     pixel_values=images,
                                     return_dict=True)
        pool_out = output.pooler_output
        # 分类头   (bsz, 768)  ->   (bsz, 3)
        final_output = self.fc(pool_out)

        loss = self.CE_Loss(final_output, labels.long()) + js_loss

        return (loss, final_output)
