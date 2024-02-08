LOC = "../../data/GroundTruth/"


class BenGRN:
    def __init__(
        do_all: bool = False,
        pert_pred: bool = False,
        binding_pred: bool = False,
        do_simulation: bool = False,
        dataset: str = "hESC",
        simulated_network=None,
    ) -> None:
        print("using " + dataset + " dataset")
        print(
            "make sure you make your prediction on this dataset in ../../data/GroundTruth/"
        )
        pass

    def __call__(self, grnadata):
        if self.dataset == "hESC":
            if grndata.X.shape != (100, 100):
                raise ValueError(
                    "The dataset provided does not match the expected shape of this ground truth.\
                        \nAre you sure you have used the hESC in ../../data/GroundTruth/ ?"
                )

        if self.do_all or self.pert_pred:
            self.pert_pred(grnadata)

        if self.do_all or self.binding_pred:
            self.binding_pred(grnadata)

        if self.do_simulation:
            if self.simulated_network is None:
                raise ValueError("Simulated network hasn't been created or given yet")

        if self.do_all or self.do_literature:
            self.litterature(grnadata)

    def generate_simulation_data(self):
        pass

    def pert_pred(self, grnadata):
        pass

    def binding_pred(self, grnadata):
        pass

    def litterature(self, grnadata):
        pass
