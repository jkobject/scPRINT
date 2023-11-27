import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union
from typing_extensions import Self, Literal
from scprint import utils
from datasets import Dataset
import lamindb as ln
import anndata as ad

additional_tissues = {
    "UBERON:0037144": "wall of heart",
    "UBERON:0003929": "digestive tract epithelium",
    "UBERON:0002020": "gray matter",
    "UBERON:0000200": "gyrus",
    "UBERON:0000101": "lobe of lung",
    "UBERON:0001981": "blood vessel",
    "UBERON:0001474": "bone element",
}


additional_diseases = {
    "MONDO:0001106": "kidney failure",
    "MONDO:0021166": "inflammatory disease",
    "MONDO:0004992": "cancer",
    "MONDO:0004994": "cardiomyopathy",
    "MONDO:0700065": "trisomy",
    "MONDO:0021042": "glioma",
    "MONDO:0005265": "inflammatory bowel disease",
    "MONDO:0005550": "infectious disease",
    "MONDO:0005059": "leukemia",
}


@dataclass
class Dataset(torchDataset):
    lamin_dataset: ln.Dataset
    organism: list[str] = ["NCBITaxon:9606", "NCBITaxon:10090"]

    def __post_init__(self):
        files = self.lamin_dataset.files.all().df()
        if (
            len(files["accessor"].unique()) != 1
            or files["accessor"].unique()[0] != "AnnData"
        ):
            raise TypeError("lamin_dataset must only contain AnnData objects")
        print(
            "won't do any check but we recommend to have your dataset coming from local storage"
        )
        print("total dataset size is {} Gb".format(files["size"].sum() / 1e9))
        print("---")
        print("grouping into one collection")

        self.anndatas = []
        for file in os.listdir(self.path):
            if file.endswith(".h5ad"):
                self.anndatas.append(
                    ad.read_h5ad(os.path.join(self.path, file), backed=True)
                )
        for i, val in enumerate(localpath):
            adata_ = ad.read_h5ad(val.path, backed=True)
            dataset_id = cx_dataset.files.all()[i].uid
            adata_.obs["dataset_id"] = dataset_id
            adata_.obs["dataset_id"] = adata_.obs["dataset_id"].astype("category")
            list_adata[dataset_id] = adata_
        # generate tree from ontologies
        self.groupings, _, self.cell_types = get_ancestry_mapping(
            set(adata.obs["cell_type_ontology_term_id"].unique()), celltypes.df()
        )
        self.groupings, _, self.cell_types = get_ancestry_mapping(
            set(adata.obs["cell_type_ontology_term_id"].unique()), celltypes.df()
        )
        self.groupings, _, self.cell_types = get_ancestry_mapping(
            set(adata.obs["cell_type_ontology_term_id"].unique()), celltypes.df()
        )
        genesdf = ln.Gene.df()
        genesdf = genesdf.drop_duplicates(subset="ensembl_gene_id")
        genesdf = genesdf.set_index("ensembl_gene_id")
        # mitochondrial genes
        genesdf["mt"] = genesdf.symbol.astype(str).str.startswith("MT-")
        # ribosomal genes
        genesdf["ribo"] = genesdf.symbol.astype(str).str.startswith(("RPS", "RPL"))
        # hemoglobin genes.
        genesdf["hb"] = genesdf.symbol.astype(str).str.contains(("^HB[^(P)]"))
        ...

    # def __len__():

    # def __getitem__(self, idx):

    def add(adata):
        if type(adata) is str:
            adata = anndata.read_h5ad(adata)

        elif type(adata) is anndata.AnnData:
            pass
        else:
            raise TypeError(
                "anndata must be either a string or an anndata.AnnData object"
            )


class GeneAnnData(AnnData):
    def __init__():
        super().__init__()

    def use_prior_network(
        self, name="collectri", organism="human", split_complexes=True
    ):
        if name == "tflink":
            TFLINK = "https://cdn.netbiol.org/tflink/download_files/TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv.gz"
            net = utils.pd_load_cached(TFLINK)
            net = net.rename(columns={"Name.TF": "regulator", "Name.Target": "target"})
        elif name == "htftarget":
            HTFTARGET = "http://bioinfo.life.hust.edu.cn/static/hTFtarget/file_download/tf-target-infomation.txt"
            net = utils.pd_load_cached(HTFTARGET)
            net = net.rename(columns={"TF": "regulator"})
        elif name == "collectri":
            import decoupler as dc

            net = dc.get_collectri(organism=organism, split_complexes=split_complexes)
            net = net.rename(columns={"source": "regulator"})
        self.add_prior_network(net)

    def add_prior_network(self, prior_network: pd.DataFrame):
        # validate the network dataframe
        required_columns: list[str] = ["target", "regulators"]
        optional_columns: list[str] = ["type", "weight"]

        for column in required_columns:
            assert (
                column in prior_network.columns
            ), f"Column '{column}' is missing in the provided network dataframe."

        for column in optional_columns:
            if column not in prior_network.columns:
                print(
                    f"Optional column '{column}' is not present in the provided network dataframe."
                )

        assert (
            prior_network["target"].dtype == "str"
        ), "Column 'target' should be of dtype 'str'."
        assert (
            prior_network["regulators"].dtype == "str"
        ), "Column 'regulators' should be of dtype 'str'."

        if "type" in prior_network.columns:
            assert (
                prior_network["type"].dtype == "str"
            ), "Column 'type' should be of dtype 'str'."

        if "weight" in prior_network.columns:
            assert (
                prior_network["weight"].dtype == "float"
            ), "Column 'weight' should be of dtype 'float'."

        # check that we match the genes in the network to the genes in the dataset

        print(
            "loaded {:.2f}% of the edges".format((len(prior_network) / init_len) * 100)
        )
        # transform it into a sparse matrix
        # add it into the anndata varp
        
        self.prior_network = prior_network
        self.network_size = len(prior_network)
        self.overla
        self.edge_freq


##########################################
################### OLD ####################
##########################################


@dataclass
class DataTable:
    """
    The data structure for a single-cell data table.
    """

    name: str
    data: Optional[Dataset] = None

    @property
    def is_loaded(self) -> bool:
        return self.data is not None and isinstance(self.data, Dataset)

    def save(
        self,
        path: Union[Path, str],
        format: Literal["json", "parquet"] = "json",
    ) -> None:
        if not self.is_loaded:
            raise ValueError("DataTable is not loaded.")

        if isinstance(path, str):
            path = Path(path)

        if format == "json":
            self.data.to_json(path)
        elif format == "parquet":
            self.data.to_parquet(path)
        else:
            raise ValueError(f"Unknown format: {format}")


@dataclass
class MetaInfo:
    """
    The data structure for meta info of a scBank data directory.
    """

    on_disk_path: Union[Path, str, None] = None
    on_disk_format: Literal["json", "parquet"] = "json"
    main_table_key: Optional[str] = None
    # TODO: use md5 to check the vocab file name on disk
    gene_vocab_md5: Optional[str] = None
    study_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of study IDs"},
    )
    cell_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of cell IDs"},
    )
    # md5: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "MD5 hash of the gene vocabulary"},
    # )

    def __post_init__(self):
        if self.on_disk_path is not None:
            self.on_disk_path: Path = Path(self.on_disk_path)

    def save(self, path: Union[Path, str, None] = None) -> None:
        """
        Save meta info to path. If path is None, will save to the same path at
        :attr:`on_disk_path`.
        """
        if path is None:
            path = self.on_disk_path

        if isinstance(path, str):
            path = Path(path)

        manifests = {
            "on_disk_format": self.on_disk_format,
            "main_data": self.main_table_key,
            "gene_vocab_md5": self.gene_vocab_md5,
        }
        with open(path / "manifest.json", "w") as f:
            json.dump(manifests, f, indent=2)

        # TODO: currently only save study table, add saving other tables
        with open(path / "studytable.json", "w") as f:
            json.dump({"study_ids": self.study_ids}, f, indent=2)

    def load(self, path: Union[Path, str, None] = None) -> None:
        """
        Load meta info from path. If path is None, will load from the same path
        at :attr:`on_disk_path`.
        """
        if path is None:
            path = self.on_disk_path

        if isinstance(path, str):
            path = Path(path)

        with open(path / "manifest.json") as f:
            manifests = json.load(f)
        self.on_disk_format = manifests["on_disk_format"]
        self.main_table_key = manifests["main_data"]
        self.gene_vocab_md5 = manifests["gene_vocab_md5"]

        if (path / "studytable.json").exists():
            with open(path / "studytable.json") as f:
                study_ids = json.load(f)
            self.study_ids = study_ids["study_ids"]

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> Self:
        """
        Create a MetaInfo object from a path.
        """
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")

        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")

        if not (path / "manifest.json").exists():
            raise ValueError(f"Path {path} does not contain manifest.json.")

        meta_info = cls()
        meta_info.on_disk_path = path
        meta_info.load(path)
        return meta_info
