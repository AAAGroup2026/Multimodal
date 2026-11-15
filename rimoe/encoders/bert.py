import torch
import torch.nn as nn

__all__ = ['BertTextEncoder']

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased', local_cache_path=None):
        super().__init__()
        from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

        models = {
            'bert': (BertModel, BertTokenizer),
            'roberta': (RobertaModel, RobertaTokenizer),
        }
        model_class, tokenizer_class = models[transformers]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained, cache_dir=local_cache_path)
        self.model = model_class.from_pretrained(pretrained, cache_dir=local_cache_path)
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]
        return last_hidden_states
