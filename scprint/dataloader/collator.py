
class Collator:

    def __init__(self, tokenizer, max_len, embeddings):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.embeddings = embeddings

    def __call__(self, batch):
        # get the embeddings
        # copy
        # do count selection
        # do encoding of counts
        # do encoding of graph location
        # find a way to encode all the nodes in some sparse way
        
    