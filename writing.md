# 10 points and ideas to push forth

single cell Transformer models are at the beginning of their development still but hold many great promises
We have developed of novel foundational model with many new features, inductive biases, and zero-shot abilities
Well-trained transformer models can perform good zero-shot GRN prediction
Head selection is important (as in ) and parameter selection too
Different GRN Ground Truths are very different in what they mean
We are lacking a good comprehensive ground truth
We can look in the meantime at some other information about the GRN…
We can look at the zero-shot performance of GRN-generating models, on related orthogonal tasks
scPRINT also performs competitively on these tasks (Denoising, Batch Effect Corr, cell type pred)
scPRINT can be used to perform a full suite of tasks with a single model although work remains on further explainability

# Potential titles: ​​What 50 million cells can tell us about gene networks?

integration of 50 million cells’ expression to uncover their regulatory mechanism
scPrint: Solving the inherent issues of current expression LLMs using gene regulation inference
scPrint: (Towards?) an explainable scRNAseq foundational model for gene network inference
scPrint: (towards?) single cell Foundation models
learn the cell mechanism
are unsupervised GRN predictors/cell modelers


## Main

### Definitions
Understanding the cellular mechanism would be considered a milestone in biology. It would allow us to better predict cell behavior and the impact of new drugs. A step towards that goal is the inference of gene networks that explain the maintenance of gene expression across cells. [] Many approaches have been developed to infer such networks from shallow cell measurements and although other multi-omics approaches can greatly complement the inference of gene networks, scRNAseq remains the most commonly available and cheapest single-cell assay.
Gene Regulatory networks can mean different things. Our goal is to consider them as a schematic representation of the cell’s internal behavior, a cell model at the level of transcripts. A gene network will thus encompass any set of genes that can impact another gene’s RNA abundance. Whether they are TFs recruiting polymerase or RNA-RNA interactions impacting the degradation of a gene. 
Many possible ground truths exist that reflect the ambiguity in gene networks. ChIP-seq, perturb-seq, ground truths based on the literature and networks that generate fake matching expression data through ODEs.
And many tools have been developed to make such Gene Network predictions: from simple models using regression to predict the expression of a gene from others, to more complex methods solving ODEs, using graph neural networks, and more. 

## Issues :
However, many of these tools often don’t scale to the number of genes present in a cell or the number of cells in large datasets. Moreover benchmarks like BeeLine and … have shown that most tools perform quite poorly and that the simplest approach often outperforms the rest.
GENIE3 has often been cited as one of the top-performing methods. It is also one of the simplest methods, doing gene expression regression using xgboost. 
It does make sense that a tool with an inductive bias toward simplicity and non-linearity is among the top-performers. The task is highly under-constrained, trying to predict matrices of 100s of millions of elements using very noisy measurements of a few thousand cells. And what we know about GRNs is scarce and these tools also have to relearn everything from scratch on each new dataset.
Overall, gene network inference is hard but there is a possibility that new very large models, trained on tens of millions of measurements might be able to help solve these difficulties.
Foundational Models for single-cell omics data have held promises to learn a model of the cell that would allow abilities across many tasks in cellular biology like cell type annotation, batch effect correction, perturbation prediction, and gene network inference.
Geneformer presented a first approach by reusing a BERT Transformer model used in language modeling and modifying a cell’s expression profile to roughly look like a long sentence. Retraining it from scratch and gaining the reliable ability to infer some gene relationships in specific cell types.
scGPT pushed the idea further, proposing, amongst other things, a novel encoding of genes and their expression, a new pretraining methodology similar to the autoregressive pretraining in language models, and a set of fine tuning losses for different tasks.  
However, recent, yet unreviewed works have shown that these models seem to require fine tuning to achieve meaningful performances on most tasks. But the pretrained models don’t seem to outperform their untrained counterparts post fine tuning. Some of these issues might come down to the pretraining task of masking 15% of the genes which might be unexpectedly easy for such large models. The models have also been pretrained on data with lots of noise and large class imbalances. Where a majority of cells come from the same set of tissues. 
Yet given a good enough training paradigm such foundation models could solve some of the gene network inference issues. Both outputting gene networks as explainable output but also generating useful output on many related tasks such as cell type prediction, batch effect correction, denoising or zero imputation, cell fate predictions, perturbation predictions and more.
Thus the best tasks to validate such a model cannot be on an almost unknown ground truth of which gene influence which other (from ChIPseq and the like) but has to be on orthogonal tasks such as the ones presented above. The actual GRN graph

## The model

We present a novel transformer models bringing novel inductive biases and pretraining strategies…

## The results

We aim to benchmark this novel foundation model on challenging gene network inference tasks. First to really look at its ability to map a meaningful model of the cell. But also because this would bring a tool for the community to make sense of these model’s predictions by giving an explainable output of their inner computation.

Since GRN ground truths are still sparse. We come up with  a compendium of tasks that a model with a good internal gene network should be able to perform well at in a zero shot fashion. They concern denoising, cell type prediction and low dimensional embedding with batch effect correction.
The code and data is available at: 
The pretraining data is available at: 
the model weight, hyperparameters and training logs are available at: 
Results
what biologists want to do, what they can do with this tool, how to do this, focus on a specific biological problem

## [INTRO FIG 1] scPRINT: a scRNAseq foundation model for gene network inference

### intro

Given the relative infancy of these large models, we have produced our own called scPRINT. It builds on top of scGPT, Universal Cell Embedding and Geneformer. While adding many novelties and inductive biases to the model and its pretraining methodologies. We show that they help the model attain good zero shot abilities on many important tasks of cellular biology.
First scPRINT is based on a bidirectional transformer with flashattention2 which helps the model generate fully connected gene x gene attention matrices at the scale of the genome. 
training

scPRINT is trained with a set of novel multi task pre-training using 50M cells from Cellxgene across multiple species, diseases and ethnicities. which represents roughly around 100 billion tokens.
Its main pretraining task is denoising. In the context of expression data, we use downsampling of transcript count per cell as noise. 0 counts being a fully random, unknown profile.
We show that this strategy performs better than masked language modeling (see supplementary fig1,2,3)
Our second pretraining task is called bottleneck learning and take inspiration from the work of naftali tishbi, AE, VAEs. We generate cell level embeddings from expression profiles and pass them back to the model without the expression information. Which requires the model to compress expression into embeddings
we make it so that the expression profile is put into multiple deconvolved embedding, each representing a specific facet of the cell state (e.g. cell type, tissue, disease, sex, sequencer, …) 
This allows for an as of yet untested ability of the model to perform counterfactual generation: mixing embeddings representing different facets of cell states (e.g. fibroblast + cancer + pancreas tissue + female) to generate novel unseen cell states.
We add label prediction as part of the pretraining. The assumption is that this is often not done during pretraining due to the relative scarcity and noisiness of labels. We believe that our hierarchical approach and the labeling requirement in cellxgene makes it a net positives, this is also what helps us deconvolve the various cell embeddings.

Cellxgene classes follow an ontology. An ontology is a powerful structure that defines relationships amongst the different labels. We train the model only on the leaf labels in the hierarchy of ontology. Thus if a label is not very specific (e.g. neuron) we let the model choose the best leaf label for it (e.g. dopaminergic neuron). This way we can generate meaningful training signals from even very coarse labels. Each class of labels uses a specific embedding.  
We not only make the code and model weights publicly available, but also the pretraining strategies and datasets. We also release a dataloader for extremely large compendium of datasets like cellxgene. It is  built on top of the lamin.ai toolkit and interfacing simply with anndata and scvi dataloaders.
inductive biases
We achieve very performant training speeds. e.g. only requiring 1 A100 GPU for 72 hours to train our smallest model, greatly reducing the barrier to entry for any computational biology lab. More information on supplementary table1.

An assumption in language modeling is that the less inductive bias, the better. However (LLM)s contain many architectural choices related to the data modality they are working on. LLMs  are also trained on roughly 100x more tokens than exist in single cell RNAseq. We believe that adding good inductive biases will be an important feature of AI models for biology.  
First each gene in a cell is converted to an embedding: It corresponds to the average of 3 different elements: 

1. an embedding representing the gene itself. Here we used the ESM2 embedding of the most common protein product of each gene as a representation of that gene while imperfect in some ways this inductive bias allows the model to learn representations that potentially apply to even unseen genes. From unseen species or which integrate specific mutations. First implemented in UCE, this provides the model information that relates to the gene product’s structure as well as ontology and similarity to other genes. This also speeds up the training greatly, particularly for small models (see table). We show that this is a great gene representation but that model performance can be increased by refining gene embeddings further during training. However we elect not to do so, to keep the model’s versatility to work on unseen genes.

2. an embedding of the gene position in the genome. This has also been proposed in UCE and helps the model understand that genes that share similar locations tend to be regulated by similar regulatory regions. This relationship is well known in cellular biology and we show that adding this information helps model performance and increase training speed.

3. an embedding of the gene expression in the cell, which amounts to a projection of the gene’s expression using a set of linear layers. GeneFormer came up with a ranking strategy based on a gene expression compared to a baseline expression and scGPT, with a similar concept which insteads uses binning of log normalized counts. 

Despite the rationale given in both papers, we haven’t found any significant downsides for using raw counts and prefer to not apply inductive biases on this part of the model since no good heuristic on count data.
Finally when encoding a cell expression profile, during pretraining only a subset of 2200 genes are used. We randomly choose 2200 expressed genes and pad them with randomly sampled unexpressed genes if less than 2200 genes are expressed. We thus let the model train on seemingly unexpressed genes. First, not using padding allows us to not waste compute on unfilled attention matrices. Moreover the input data is very noisy. Some genes are true zero but many, half are likely due to dropout as we expect around half the genome to be expressed at any given time in a cell. We thus let the model learn about the data distribution and the presence of dropout even in the ground truth data.
This is modeled in the output of the model where we reuse the scVI graphical model. Effectively the model outputs the parameters of a probability distribution of the counts. Here we use a zero inflated negative binomial to give the most expressivity to our output. The loss is then computed as the log likelihood of the gene expression given the distribution.

more in table 1

### multiple outputs

As we have seen, the model is able to generate many different outputs at inference time, without any need for fine tuning and across any single cell RNA seq-like cellular profile of multiple mammalian species. scPrint is able to do this at the scale of atlases of cells. here for 2M random cells in cellxgene  
but In addition to denoising, label prediction and embedding, one important output of scPRINT is its cell specific gene network. Following a similar approach to ESM2 we leverage the fact that transformer models compute input element by input element weighted matrices called attention matrices to generate cell level gene networks by combining them. 

With our approach the gene networks can be computed on anywhere from 1 to 100,000s cells, at the scale of the genome with commodity hardware. We believe to be amongst the first models to do so.
Finally this gene network matrix can be fine tuned to reflect the different perspective on gene networks given a ground truth gene network.
We believe that although these have been mentioned a lot in the foundation model literature, they haven’t been investigated enough and could serve as a means to understand the model’s fundamental understanding of a cell. We will now focus on this output and show that scPRINT GRNs contain relevant biological features and connections.

## [omnipath FIG 2] the Gene Networks extracted from scPRINT recover expected biological features

### presentation

For our first task we benchmark scPRINT’s gene network generated on a random test dataset of …. from cellxgene where for each cell type 2048 random cells are passed to each gene network inference model.
We compare to scGPT human pretraining model. Although scGPT’s code mentions GRN inference only by using perturb-seq data. We reapply the same method without the perturbation-baseline comparison. Effectively only using the mean of the attention matrices across cells and the last 4 attention heads in the model.
We also use GENIE3 as a reference model. GENIE3’s connection are given by applying xgboost to predict each gene’s expression given a sparse set of other genes. The prediction coefficients will determine the weight of the network. However, GENIE3 is supposed to be used by only selecting amongst TFs, biasing the model to generate TF-gene graph. This is often referred to as a Gene Regulatory Network, given the importance of TF to the regulation of gene expression. 

### results

We aim to show in these benchmarks that the recovered networks contain meaningful biological knowledge.
First we look at how much information from omnipath is contained in the network. Omnipath contains curated gene - gene connections mainly from the literature. These connections are cell type agnostic and around half are TF - gene. 
We do not expect most connections to be present in the cell type’s gene network as half of the genes will not even be expressed and many connections in omnipath might be only true in some cellular contexts. Moreover we do not expect most connections in our generated cell type specific network as omnipath only contains a tiny fraction of all true gene-gene connections.
On this benchmark we look at AUC and EPR, two metrics often used in GRN benchmarks (beeline) (see methods), where we view our task as a binary classification of connections or not on all gene-gene pairs.
AUC shows the area under the curve defined by the different precision, recall values as we increase the cutoff on the directed weighted graph predicted by our different tools.
EPR corresponds to the expected precision recall for a cutoff that leaves as many connections in the ground 


We focus more on TFs compared to GENIE3
EPR corresponds to the expected precision recall for a cutoff that leaves as many connections in the ground truth as in the prediction. it is in the form 

$EPR = \frac{2 \cdot precision \cdot recall}{precision + recall}$. 

scPRINT outperforms GENIE3 and scGPT on this benchmark. to compare ourselves to GENIE3-TF we also implement a "GRN" version of our network where we only keep TF-gene connections

We focus more on TFs compared to GENIE3 

We more often find cell type marker enrichment than GENIE3 and much more than scGPT (likely helped by classification task)
scGPT does really well at TF target enrichment. We confirm that doing masked language modeling in our model also led to great results on this assessment.


### what to say

choice of attention head is very important (because…)
hard to know what a test dataset for scGPT
we
We have seen however that, especially for small ground truths EPR can have a high variance in the range 0.8-1.4 has high variance in general


## [gwps / sroy FIG3] scPrint beats GENIE3 and scGPT on two cell -type-specific experimental ground truths

### presentation

Although we have shown that our networks represent meaningful biology. We would want validation on larger ground truth from cell type specific networks. Here we use two different benchmarking modalities.
The first one comes from sushmita roy et. al. ChIP sequencing allows biologists to peak at 
perturb-seq is a novel modality that allows biologist to interrogate the effect of   

present chipseq 

present perturb seq

present an idea of why this make sense and some of the caveats

present the second dataset

### results

we perform really well on this. this is great because this is by far the most complete dataset
what to say
issue with both modality and general issue in such dataset. (work remains on the data generation side) (comparison to ESM2)

## [other tasks FIG4] scPRINT also shows convincing performances on orthogonal tasks to GRN inference

### introduction

In the meantime, we believe that as expressed in … being able to have good performances in the orthogonal task

### denoising

Taking on the idea of …. having a good gene network should help a model denoise an expression profile by relying on known gene interactions.
As seen in FIG2 and expressed in  …. Gene networks should contain core nodes that define the cell’s state and should thus overlap with known cell type markers. A good gene network should thus help a model predict cell type labels. 

### classification of cell type

We also estimate that batch effect is a component that is independent on the cell’s gene network and thus a model with reliable gene network prediction should be able to reduce the dimensionality of an expression profile while removing the batch effects.
Although these tasks might be performed directly from the generated gene network extracted from the transformer performing these tasks, using the transformer model’s output directly will lead to better results.
over perform when taking into account that it doesn’t use any batch information and hasn’t been trained on the dataset

### batch effect correction

The only model that hasn’t been trained on the dataset. Best model for finding rare cell types. Can leverage cell type information across species. trained to predict other labels and achieves …… on the validation set. Best model
Can’t compare that its predictions are over 400 cell type whereas other model’s predictions are over 10.
likely quite bad at denoising on low cell count and on transitional developmental cell states
over perform when taking in account that it

## [Use case FIG5] scPRINT as a central tool for discovery in a bio informatics data analysis pipeline

### intro and goal

presenting the goal. We might gather dataset, fresh from the sequencer or from some GEO data source. We might not have or trust its annotation and would want a consistent embedding and annotation

scPRINT can do that. taking raw expression data and outputting rich annotations and matching sets of embeddings
batch effect correction
labels prediction

one might be interested in some rare cell states. Those have few cells and extracting information can be difficult. they are also often the most interesting. responsible for a breadth of disease and 
based on the available transcriptomic profile and inferred  

## Discussion

### output

We have presented a resource for working with and validating Gene Regulatory Networks infered from single cell RNA sequencing. Moreover we have built a novel single cell RNA sequencing foundational model trained on 50 million single cell profiles pan tissue, disease and species.
Although it hasn’t been trained for it, this model generates gene networks from its predictions which can be seen as a way to better understand its prediction
We have also presented a dataloader called scDataloader, built on top of laminDB and cellxgene that allows the first reproducible and open source pretraining of such foundation models.
Issues & next step
However we acknowledge that much work remains to be done on these models from their abilities to generate graphs, their explanability, as well as the breadth of task they can undertake. We are still exploring the important question of time point data and perturbation predictions. Many ideas remain un explored on the pretraining tasks and the ways to integrate other omics data to such models.
Transcription is also much more complex than and in the future only novel data modalities will be able to solve the gap remaining to model cell behavior. Mostly we believe that there is a need to better sample the dark transcriptome.
We would like to thank additional collaborators such as laminDB and my lab members
Moreover we would like to acknowledge the very important contributions without which this work could not have been done: geneformer, UCE, scGPT, scVI, tri dao, pytorch, lightning, for their ideas and open source code upon which a lot of scPRINT is based. omnipath, scenic+, open problems and sushmita roy for their ground truths and benchmarking tools.

## Methods

### Denoising
We view denoising as taking a cell profile xi = .. 
that has a total count sum xi = ti
we add a certain amount of noise by downsampling it to..

### Graphical model
scPRINT’s transformer model outputs the parameters of a zero inflated negative binomial function.
Based on the work of (counts) zero inflation is the best distribution for taking in account a broad range of transcriptomic measurement where some have enough dropout that a zero inflation term is needed to model it
One can see a negative binomial distribution:
…
as a mixture of poisson
…
and a zero inflated negative binomial as just 

in our case and similarly to scVI we define our ZINB as: …
with a change of parameters

However we add that instead of being a fixed parameter learnt across the dataset, the over dispersion term .. is also generated from the counts
effectively the model learns that the dispersion might be greater not only depending on the gene but also on the sequencer, the cell type, and the sequencing depth.
gene encoder

### Bottleneck learning

naftali tishbi
Classification as a pretraining task

Hierarchical Classifier

scDataloader
lamin.ai
preprocessing
h5ads
random weighted sampling
lightning datamodule
extracting meta-cell gene networks from attention matrices
attention matrices equations
mutli head per layer
meta cell level equations
doing classification
Gene networks from genome-wide perturb seq
cutoff on diff expr of cells with KO A vs baseline
for all diff expr
giving a gene x gene directed weighted graph
BenGRN metrics on gene networks
EPR and AUC, enrichment

Availabability & Access
code on
pretraining on
model weights on
pre-training logs on
data on
dataset on
other datasets use on
GrnnData
BenGRN
scDataLoader
Supplementary figures
[FIG S1] model size comparison
[FIG S2] model performance increase on omnipath
untrained, small, medium, large, vlarge, scGPT
[FIG S3] model performance increase on classification
[FIG S4] model performance increase on embedding task
[FIG S5] ablation study ( impact on model performance at…)
[FIG S6] overlap of different GRN ground truths
saying that we don't expect a lot of overlap in any case
[FIG S7] diff expr analysis on naive B cell vs rest

[FIG S8] schematic representation of the hierarchical classifier

[FIG S9] schematic representation of the graphical model


[Table 1] list of novelties in scPRINT and comparison to scGPT, GeneFormer & UCE.

