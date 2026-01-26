from torch import nn
from .modeling_unimo import UnimoModel
from torch.nn import CrossEntropyLoss


class UnimoModelF(nn.Module):
    def __init__(self, args, vision_config, text_config):
        super(UnimoModelF, self).__init__()
        self.args = args
        self.vision_config = vision_config
        self.text_config = text_config
        self.model = UnimoModel(args, vision_config, text_config)
        self.fc = nn.Linear(self.text_config.hidden_size, 3)
        self.CE_Loss = CrossEntropyLoss()
        self.last_pool_out = None

    def forward(self, input_ids, attention_mask, token_type_ids, labels, images):
        output, js_loss = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     pixel_values=images, return_dict=True)
        pool_out = output.pooler_output  # 维度: (64, 768)
        self.last_pool_out = pool_out

        # 分类头   (bsz, 768)  ->   (bsz, 3)
        final_output = self.fc(pool_out)

        loss = self.CE_Loss(final_output, labels.long()) + js_loss

        # js_loss 消融
        # loss = self.CE_Loss(final_output, labels.long())

        return (loss, final_output)
