    def use_prior_network(
        self, name="collectri", organism="human", split_complexes=True
    ):
        """
        use_prior_network loads a prior GRN from a list of available networks.

        Args:
            name (str, optional): name of the network to load. Defaults to "collectri".
            organism (str, optional): organism to load the network for. Defaults to "human".
            split_complexes (bool, optional): whether to split complexes into individual genes. Defaults to True.

        Raises:
            ValueError: if the provided name is not amongst the available names.
        """
        # TODO: use omnipath instead
        if name == "tflink":
            TFLINK = "https://cdn.netbiol.org/tflink/download_files/TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv.gz"
            net = pd_load_cached(TFLINK)
            net = net.rename(
                columns={"Name.TF": "regulator", "Name.Target": "target"}
            )
        elif name == "htftarget":
            HTFTARGET = "http://bioinfo.life.hust.edu.cn/static/hTFtarget/file_download/tf-target-infomation.txt"
            net = pd_load_cached(HTFTARGET)
            net = net.rename(columns={"TF": "regulator"})
        elif name == "collectri":
            import decoupler as dc

            net = dc.get_collectri(
                organism=organism, split_complexes=split_complexes
            )
            net = net.rename(columns={"source": "regulator"})
        else:
            raise ValueError(
                f"provided name: '{name}' is not amongst the available names."
            )
        self.add_prior_network(net)

    def add_prior_network(self, prior_network: pd.DataFrame, init_len):
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

        # TODO: check that we match the genes in the network to the genes in the dataset

        print(
            "loaded {:.2f}% of the edges".format(
                (len(prior_network) / init_len) * 100
            )
        )
        # TODO: transform it into a sparse matrix
        self.prior_network = prior_network
        self.network_size = len(prior_network)
        # self.overlap =
        # self.edge_freq