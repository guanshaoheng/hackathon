# Problem Statement: AI-Aided Genome Fragment Ordering for Post-Assembly Curation

# Background

In genome assembly, the Hi-C map is a widely used method to provide structural information and aid in arranging genomic fragments. However, after initial scaffolding, some genomic fragments may still be misjoined, incorrectly oriented, or misplaced. This issue complicates the final assembly process and requires extensive manual curation. The curation and review process, performed by specialized teams such as the team in the Tree of Life (ToL) project at Sanger Institute, involves arranging these fragments into their correct order and orientation before the assembly can be finalized and submitted to public databases. This process is labor-intensive for genomes with numerous small fragments, coupled with large number of samples awaiting curation.

# Objective

To streamline the curation process, we aim to leverage AI models to predict the correct order and orientation of shuffled genomic fragments in a Hi-C map. Specifically, the challenge is to train a graph neural network (GNN) that can learn from a set of training samples, where both the shuffled and correct orders are provided, and use this model to predict the correct order for new shuffled samples.

# Problem Description

The input data consists of high-resolution Hi-C contact maps, represented as matrices with dimensions of `32,768 x 32,768` pixels. Each map is divided into smaller fragments that are then shuffled, representing a disordered arrangement of genomic segments. You have:

- 300 training samples: Each sample includes a shuffled Hi-C map and its corresponding correct order.
- 100 test samples: Each sample includes a shuffled Hi-C map without the correct order.

The goal is to train a GNN (or use any other method) that can accurately model the relationships among the fragments in these maps and predict a joint likelihood matrix indicating the probability of each fragment being correctly ordered. This matrix will be used to reorder the fragments, minimizing the likelihood of mis-joins, inversions, and misplacements.

# Challenges

- High Dimensionality: The Hi-C maps are large (`32,768 x 32,768` pixels), posing a challenge for computational efficiency.
- Complex Fragment Relationships: Fragment mis-joins and inversions add complexity to the task, making simple reordering methods ineffective.
- Data Scarcity for Testing: Only 300 labeled samples are available for training, with 100 samples to be predicted, making generalization critical.
- Graph Modeling: Representing Hi-C maps as graphs and defining meaningful relationships between fragments are key for training an effective GNN.

# Proposed Solution

We propose to develop a GNN-based model to learn the joint likelihood among shuffled fragments, leveraging the structural information present in the Hi-C maps. The model will output a likelihood matrix, which will then be used to predict the correct order and orientation of fragments. The solution will be evaluated based on the accuracy of reordering the fragments, reducing the curation workload for the team, and improving the overall quality of genome assembly submissions.


# Dataset 

For dataset, please check the [link](https://drive.google.com/drive/folders/1D5fP0aoTjvanmRMuxK1dXl7p3HAFhzIf?usp=drive_link)