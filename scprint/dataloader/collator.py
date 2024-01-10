
class Collator:

    def __init__(self, tokenizer, max_len, embeddings):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.embeddings = embeddings

    def __call__(self, batch):
        # do count selection
        # do encoding of counts

        # find the associated gene ids (given the species)
        # get the NN cells

        # do encoding / selection a la scGPT
        
        # do encoding of graph location
        # encode all the edges in some sparse way
        
    