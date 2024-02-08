from scprint import data_utils\
import lamindb as ln\
import lnschema_bionty as lb\
lb.settings.organism = "human"\
cx_dataset = ln.Collection.using("laminlabs/cellxgene").one()\
mydataset = data_utils.load_dataset_local(lb, cx_dataset, "~/scprint/", name="cellxgene-local", description="the full cellxgene database", only=(230,800))