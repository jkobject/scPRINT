import pdb
from typing import Dict, Optional, Union, Callable

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix
from uuid import uuid4

from scprint.dataset import utils as data_utils
from scprint import logger

import lamindb as ln
import bionty as bt

FULL_LENGTH_ASSAYS = [
    "EFO: 0700016",
    "EFO:0008930",
    "EFO:0008931",
]


class Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        lb,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
        length_normalize: bool = False,
        additional_preprocess: Optional[Callable[[AnnData], AnnData]] = None,
        additional_postprocess: Optional[Callable[[AnnData], AnnData]] = None,
        force_preprocess=False,
        min_dataset_size=100,
        min_valid_genes_id=10_000,
        min_nnz_genes=200,
        maxdropamount=2,
        madoutlier=5,
        pct_mt_outlier=8,
        batch_key=None,
        erase_prev_dataset: bool = False,
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        result_normed_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log1p (:class:`bool`, default: ``True``):
            Whether to apply log1p transform to the normalized data.
        result_log1p_key (:class:`str`, default: ``"X_log1p"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        result_binned_key (:class:`str`, default: ``"X_binned"``):
            The key of :class:`~anndata.AnnData` to store the binned data.
        """
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.binning = binning
        self.result_binned_key = result_binned_key
        self.additional_preprocess = additional_preprocess
        self.additional_postprocess = additional_postprocess
        self.force_preprocess = force_preprocess
        self.lb = lb
        self.min_dataset_size = min_dataset_size
        self.min_valid_genes_id = min_valid_genes_id
        self.min_nnz_genes = min_nnz_genes
        self.maxdropamount = maxdropamount
        self.madoutlier = madoutlier
        self.pct_mt_outlier = pct_mt_outlier
        self.batch_key = batch_key
        self.erase_prev_dataset = erase_prev_dataset
        self.length_normalize = length_normalize

    def __call__(
        self,
        data: Union[ln.Dataset, AnnData] = None,
        name="preprocessed dataset",
        description="preprocessed dataset using scprint",
        start_at=0,
    ):
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        files = []
        if isinstance(data, AnnData):
            return self.preprocess(data)
        elif isinstance(data, ln.Dataset):
            for i, file in enumerate(data.files.all()[start_at:]):
                # use the counts matrix
                print(i)
                adata = file.load(stream=True)
                print(adata)
                try:
                    adata = self.preprocess(adata)

                except ValueError as v:
                    if v.args[0].startswith(
                        "Dataset dropped because contains too many secondary"
                    ):
                        print(v)
                        continue
                    else:
                        raise v
                if self.erase_prev_dataset:
                    adata.write_h5ad(file.storage)
                    del adata
                    file = ln.File(
                        file.storage,
                        is_new_version_of=file,
                        description="preprocessed by scprint",
                    )
                else:
                    file = ln.File(
                        adata,
                        is_new_version_of=file,
                        description="preprocessed by scprint",
                    )
                file.save()
                files.append(file)
            dataset = ln.Dataset(files, name=name, description=description)
            dataset.save()
            return dataset
        else:
            raise ValueError("Please provide either anndata or ln.Dataset")

    def preprocess(self, adata: AnnData):
        adata = adata.to_memory()
        if self.additional_preprocess is not None:
            adata = self.additional_preprocess(adata)
        if adata.raw is not None:
            adata.X = adata.raw.X
            del adata.raw
        if adata.layers is not None:
            del adata.layers
        # check that it is a count
        if (
            int(adata.X[:100].max()) != adata.X[:100].max()
            and not self.force_preprocess
        ):  # check if likely raw data
            raise ValueError(
                "Data is not raw counts, please check layers, find raw data, or bypass with force_preprocess"
            )
            # please check layers
            # if not available count drop
        # # cleanup and dropping low expressed genes and unexpressed cells
        prevsize = adata.shape[0]
        adata.obs["nnz"] = np.array(np.sum(adata.X != 0, axis=1).flatten())[0]
        adata = adata[
            (adata.obs["nnz"] > self.min_nnz_genes)
            # or if slide-seq
            | (
                (adata.obs.assay_ontology_term_id == "EFO:0030062")
                & (adata.obs["nnz"] > (self.min_nnz_genes / 3))
            )
        ]
        if self.filter_gene_by_counts:
            sc.pp.filter_genes(adata, min_counts=self.filter_gene_by_counts)
        if self.filter_cell_by_counts:
            sc.pp.filter_cells(adata, min_counts=self.filter_cell_by_counts)
        # if lost > 50% of the dataset, drop dataset
        # load the genes
        genesdf = self.load_genes(adata.obs.organism_ontology_term_id[0])

        if prevsize / adata.shape[0] > self.maxdropamount:
            raise Exception(
                "Dataset dropped due to low expressed genes and unexpressed cells: factor of "
                + str(prevsize / adata.shape[0])
            )
        if adata.shape[0] < self.min_dataset_size:
            raise Exception(
                "Dataset dropped due to low expressed genes and unexpressed cells: current size: "
                + str(adata.shape[0])
            )
        adata = adata[adata.obs.is_primary_data]
        if adata.shape[0] < self.min_dataset_size:
            raise ValueError(
                "Dataset dropped because contains too many secondary cells"
            )

        # create random ids for all cells
        adata.obs.index = [uuid4() for _ in range(adata.shape[0])]
        intersect_genes = set(adata.var.index).intersection(set(genesdf.index))
        print(f"Removed {len(adata.var.index) - len(intersect_genes)} genes.")
        if len(intersect_genes) < self.min_valid_genes_id:
            raise Exception("Dataset dropped due to too many genes not mapping to it")
        adata = adata[:, list(intersect_genes)]
        # marking unseen genes
        unseen = set(genesdf.index) - set(adata.var.index)
        # adding them to adata
        emptyda = ad.AnnData(
            csr_matrix((adata.shape[0], len(unseen))),
            var=pd.DataFrame(index=list(unseen)),
            obs=pd.DataFrame(index=adata.obs.index),
        )
        adata = ad.concat([adata, emptyda], axis=1, join="outer", merge="only")
        # do a validation function
        adata.uns["unseen_genes"] = list(unseen)
        data_utils.validate(
            adata, self.lb, organism=adata.obs.organism_ontology_term_id[0]
        )
        # length normalization
        if (
            adata.obs["assay_ontology_term_id"].isin(FULL_LENGTH_ASSAYS).any()
            and self.length_normalize
        ):
            subadata = data_utils.length_normalize(
                adata[adata.obs["assay_ontology_term_id"].isin(FULL_LENGTH_ASSAYS)],
            )

            adata = ad.concat(
                [
                    adata[
                        ~adata.obs["assay_ontology_term_id"].isin(FULL_LENGTH_ASSAYS)
                    ],
                    subadata,
                ],
                axis=0,
                join="outer",
                merge="only",
            )
        # step 3: normalize total
        if self.normalize_total:
            sc.pp.normalize_total(adata, target_sum=self.normalize_total)
        if self.log1p and not is_log1p(adata):
            sc.pp.log1p(adata)

        # QC
        adata.var[genesdf.columns] = genesdf.loc[adata.var.index]
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20]
        )

        adata.obs["outlier"] = (
            data_utils.is_outlier(adata, "total_counts", self.madoutlier)
            | data_utils.is_outlier(adata, "n_genes_by_counts", self.madoutlier)
            | data_utils.is_outlier(
                adata, "pct_counts_in_top_20_genes", self.madoutlier
            )
        )

        adata.obs["mt_outlier"] = data_utils.is_outlier(adata, "pct_counts_mt", 3) | (
            adata.obs["pct_counts_mt"] > self.pct_mt_outlier
        )
        total_outliers = (adata.obs["outlier"] | adata.obs["mt_outlier"]).sum()
        total_cells = adata.shape[0]
        percentage_outliers = (total_outliers / total_cells) * 100
        print(
            f"Seeing {total_outliers} outliers ({percentage_outliers:.2f}% of total dataset):"
        )
        # if percentage_outliers > 50:
        #    raise Exception("More than 50% of the dataset has been dropped due to outliers.")
        # adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
        # remaining
        # step 5: subset hvg
        if self.subset_hvg:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.subset_hvg,
                batch_key=self.batch_key,
                flavor=self.hvg_flavor,
                subset=True,
            )
        for val in list(adata.obsm.keys()):
            del adata.obsm[val]
        # based on the topometry paper https://www.biorxiv.org/content/10.1101/2022.03.14.484134v2
        sc.pp.pca(adata, n_comps=500)
        sc.pp.neighbors(adata, use_rep="X_pca")
        sc.tl.leiden(adata, key_added="leiden_3", resolution=3.0)
        sc.tl.leiden(adata, key_added="leiden_2", resolution=2.0)
        sc.tl.leiden(adata, key_added="leiden_1", resolution=1.0)
        sc.tl.umap(adata)
        # additional
        if self.additional_postprocess is not None:
            adata = self.additional_postprocess(adata)

        # step 6: binning
        if self.binning:
            print("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            # NOTE: the first bin is always a spectial for zero
            n_bins = self.binning
            binned_rows = []
            bin_edges = []

            if adata.X.min() < 0:
                raise ValueError(
                    f"Assuming non-negative data, but got min value {adata.X.min()}."
                )
            for row in adata.X:
                if row.max() == 0:
                    print(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use the `filter_cell_by_counts` "
                        "arg to filter out all zero rows."
                    )
                    binned_rows.append(np.zeros_like(row, dtype=np.int64))
                    bin_edges.append(np.array([0] * n_bins))
                    continue
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)
        return adata

    def load_genes(self, organism):
        genesdf = bt.Gene(
            organism=self.lb.Organism.filter(ontology_id=organism).first().name
        ).df()
        genesdf = genesdf.drop_duplicates(subset="ensembl_gene_id")
        genesdf = genesdf.set_index("ensembl_gene_id")
        # mitochondrial genes
        genesdf["mt"] = genesdf.symbol.astype(str).str.startswith("MT-")
        # ribosomal genes
        genesdf["ribo"] = genesdf.symbol.astype(str).str.startswith(("RPS", "RPL"))
        # hemoglobin genes.
        genesdf["hb"] = genesdf.symbol.astype(str).str.contains(("^HB[^(P)]"))
        return genesdf


def is_log1p(adata: AnnData) -> bool:
    """
    Check if the data is already log1p transformed.

    Args:

    adata (:class:`AnnData`):
        The :class:`AnnData` object to preprocess.
    obs_key (:class:`str`, optional):
        The key of :class:`AnnData.obs` to use for batch information. This arg
        is used in the highly variable gene selection step.
    """
    max_, min_ = adata.X.max(), adata.X.min()
    if max_ > 30:
        return False
    if min_ < 0:
        return False

    non_zero_min = adata.X[adata.X > 0].min()
    if non_zero_min >= 1:
        return False

    return True


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(row: np.ndarray, n_bins: int) -> np.ndarray:
    """Binning the row into n_bins."""
    # TODO: use torch.quantile and torch.bucketize
    dtype = row.dtype
    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return binned_row.astype(dtype)
