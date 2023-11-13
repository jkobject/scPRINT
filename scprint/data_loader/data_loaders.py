from torchvision import datasets, transforms
from base import BaseDataLoader
from collections import Counter
import owlready2
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)




class CellxGenePreDataLoader():

    # TODO: put in config
    batch = ['self_reported_ethnicity_ontology_term_id', 'sex_ontology_term_id', 'assay_ontology_term_id', 'dataset_id']
    features = ['tissue_ontology_term_id', 'cell_type_ontology_term_id', 'development_stage_ontology_term_id', 'disease_ontology_term_id']

    def __init__(self, census, organism='all', MIN_CELLS=1000):
        if organism=="all":

        else:
            census["census_data"][organism]

        self.df = 
        self.MIN_CELLS = MIN_CELLS

    def load(self):
        list_of_datasets = {k: val for k, val in Counter(self.df['dataset_id']).items() if val > self.MIN_CELLS}

        # compute weightings
        # weight on large dev stage status, cell type tissue, disease, assay


        # get list of all elements in each category


        # make an anndata of the rare cells


        # make an anndata of the avg profile per metacell


        # make an anndata of the avg profile per assay


        # 


class adataPreProcessor():
    def __init__(self, adata):
        self.adata = adata

    def process(self):
        # do value check
        # do 

"https://stringdb-downloads.org/download/protein.physical.links.detailed.v12.0/9606.protein.physical.links.detailed.v12.0.txt.gz"

@dataclass
class CellxGeneDataLoader(BaseDataLoader):

    #TODO: create higher level groupings for CxG's tissues and dev stages

    def __init__(self, census, organism):





def adata_preprocessor(adata):



