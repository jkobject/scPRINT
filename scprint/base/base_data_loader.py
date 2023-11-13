import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from anndata.experimental import AnnLoader

# TODO: put in config
COARSE_TISSUE = {
    "adipose tissue": "",
    "bladder organ": "",
    "blood": "",
    "bone marrow": "",
    "brain": "",
    "breast": "",
    "esophagus": "",
    "eye": "",
    "embryo": "",
    "fallopian tube": "",
    "gall bladder": "",
    "heart": "",
    "intestine": "",
    "kidney": "",
    "liver": "",
    "lung": "",
    "lymph node": "",
    "musculature of body": "",
    "nose": "",
    "ovary": "",
    "pancreas": "",
    "placenta": "",
    "skin of body": "",
    "spinal cord": "",
    "spleen": "",
    "stomach": "",
    "thymus": "",
    "thyroid gland": "",
    "tongue": "",
    "uterus": "",
}

COARSE_ANCESTRY = {
    "African": "",
    "Chinese": "",
    "East Asian": "",
    "Eskimo": "",
    "European": "",
    "Greater Middle Eastern  (Middle Eastern, North African or Persian)": "",
    "Hispanic or Latin American": "",
    "Native American": "",
    "Oceanian": "",
    "South Asian": "",
}

COARSE_DEVELOPMENT_STAGE = {
    "Embryonic human": "",
    "Fetal": "",
    "Immature": "",
    "Mature": "",
}

COARSE_ASSAY = {
    "10x 3'": "",
    "10x 5'": "",
    "10x multiome": "",
    "CEL-seq2": "",
    "Drop-seq": "",
    "GEXSCOPE technology": "",
    "inDrop": "",
    "microwell-seq": "",
    "sci-Plex": "",
    "sci-RNA-seq": "",
    "Seq-Well": "",
    "Slide-seq": "",
    "Smart-seq": "",
    "SPLiT-seq": "",
    "TruDrop": "",
    "Visium Spatial Gene Expression": "",
}
CELL_ONTO: str = "https://github.com/obophenotype/cell-ontology/releases/latest/download/cl-basic.owl"
TISSUE_ONTO: str = "https://github.com/obophenotype/uberon/releases/latest/download/uberon-basic.owl"
ANCESTRY_ONTO: str = "https://raw.githubusercontent.com/EBISPOT/hancestro/main/hancestro-base.owl"
ASSAY_ONTO: str = "https://github.com/obophenotype/uberon/releases/latest/download/uberon-basic.owl"
DEVELOPMENT_STAGE_ONTO: str = "http://purl.obolibrary.org/obo/hsapdv.owl"
DISEASE_ONTO: str = 'https://raw.githubusercontent.com/EBISPOT/efo/master/efo-base.owl'


class BaseDataLoader(AnnLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        
        ontology_dn = owlready2.get_ontology(DEVELOPMENT_STAGE_ONTO)
        ontology_dn.load()
        ontology_cl = owlready2.get_ontology(CELL_ONTO)
        ontology_cl.load()
        ontology_ub = owlready2.get_ontology(TISSUE_ONTO)
        ontology_ub.load()
        ontology_ac = owlready2.get_ontology(ANCESTRY_ONTO)
        ontology_ac.load()
        ontology_di = owlready2.get_ontology(DISEASE_ONTO)
        ontology_di.load()
        
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


    def use_prior_network(self, name="collectri", organism='human', split_complexes=True):
        if name == "tflink":
            TFLINK = "https://cdn.netbiol.org/tflink/download_files/TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv.gz"
            net = utils.pd_load_cached(TFLINK)
            net = net.rename(columns={"Name.TF": 'regulator', 'Name.Target': "target"})
        elif name == "htftarget":
            HTFTARGET = "http://bioinfo.life.hust.edu.cn/static/hTFtarget/file_download/tf-target-infomation.txt"
            net = utils.pd_load_cached(HTFTARGET)
            net = net.rename(columns={'TF': 'regulator'})
        elif name == 'collectri'
            import decoupler as dc
            net = dc.get_collectri(organism=organism, split_complexes=split_complexes)
            net = net.rename(columns={'source': 'regulator'})
        self.add_prior_network(net)

    def add_prior_network(self, prior_network: pd.DataFrame):
        # validate the network dataframe
        required_columns: list[str] = ["target", "regulators"]
        optional_columns: list[str] = ["type", "weight"]

        for column in required_columns:
            assert column in prior_network.columns, f"Column '{column}' is missing in the provided network dataframe."

        for column in optional_columns:
            if column not in prior_network.columns:
                print(f"Optional column '{column}' is not present in the provided network dataframe.")

        assert prior_network['target'].dtype == 'str', "Column 'target' should be of dtype 'str'."
        assert prior_network['regulators'].dtype == 'str', "Column 'regulators' should be of dtype 'str'."

        if 'type' in prior_network.columns:
            assert prior_network['type'].dtype == 'str', "Column 'type' should be of dtype 'str'."

        if 'weight' in prior_network.columns:
            assert prior_network['weight'].dtype == 'float', "Column 'weight' should be of dtype 'float'."

        # check that we match the genes in the network to the genes in the dataset

        print("loaded {:.2f}% of the edges".format((len(prior_network)/init_len)*100))


    def get_molecular_embeddings():
        pass


def get_ancestry_mapping(url, type="cell_type"):
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    tot = set()
    ntot = set()
    for i, j in data.items():
        tot |= set(j)
        ntot |= set([i])
    # for each cell type, get all its ancestors
    ancestors = {}
    for val in tot | ntot:
        if type=="cell_type":
            ancestors[val] = set(_ancestors(val)) - set([val, 'Thing'])
        elif type=="tissue":
            ancestors[val] = set(_ancestors_ti(val)) - set([val, 'Thing'])
        elif type=="ancestry":
            ancestors[val] = set(_ancestors_ac(val)) - set([val, 'Thing'])
        else:
            raise ValueError("type must be 'cell_type' or 'tissue'")
    full_ancestors = set()
    for val in ancestors.values():
        full_ancestors |= set(val)

    # remove the things that are not in CxG
    full_ancestors = full_ancestors & set(ancestors.keys())

    # if a cell type is not an ancestor then it is a leaf
    leafs = tot - full_ancestors
    full_ancestors = full_ancestors - leafs
    # for each ancestor, make a dict of groupings of leafs that predict it
    groupings = {}
    for val in full_ancestors:
        groupings[val] = set()
    for leaf in leafs:
        for ancestor in ancestors[leaf]:
            if ancestor in full_ancestors:
                groupings[ancestor].add(leaf)

    return groupings, full_ancestors, leafs


@lru_cache(maxsize=None)
def _ancestors(entity_value, entity_type, onto):
    ancestors = set()
    entity_iri = entity_value.replace(":", "_")
    entity = onto.search_one(iri=f"http://purl.obolibrary.org/obo/{entity_iri}")
    if entity_type == 'dv':
        for val in entity.ancestors(include_constructs = True, include_self = False):
            try:
                ancestors.add(val.name)
            except AttributeError:
                ancestors.add(val.value.name)
                ancestors |= _ancestors(entity_type, val.value.name)
        print(len(ancestors))
    else:
        ancestors = (
            [i.name.replace("_", ":") for i in entity.ancestors()]
            if entity
            else [entity_value]
        )
    return ancestors