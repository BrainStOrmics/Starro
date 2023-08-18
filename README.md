## Starro: a uniform framework of cell segmentation on spatially resolved transcriptomes

[![python~=3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/)
[![documentation](https://readthedocs.org/projects/spateo-release/badge/?version=latest)](https://spateo-release.readthedocs.io/en/latest/tutorials/notebooks/cell_segmentation.html)

[Quick Example](https://github.com/Bai-Lab/Starro/blob/main/notebooks/starro_rna_seg_tutorial.ipynb) - [Citation](https://www.biorxiv.org/content/10.1101/2022.12.07.519417v1)

![image](https://github.com/Bai-Lab/Starro/assets/37856906/603e3bf3-0bd9-4633-938f-9ec17c76e22c)

<p align="justify">
Cell segmentation, the process of distinguishing boundaries of individual cells, is an essential prerequisite for analyses of emerging subcellular-resolution sequencing-based spatial transcriptomics (sST). Conventional image segmentation techniques rely on <i>in situ</i> cell staining, resulting in multiple technical challenges during subsequent <i>ex situ</i> sequencing process in sST experiments. Here, we present Starro, a segmentation method that directly leverages RNA signals for precise delineation of cellular boundaries from sST data. We demonstrate Starro’s robustness and accuracy across extensive benchmarks, including both simulated and real data from sequencing-based and imaging-based ST data. Additionally, we showcase how Starro empowers various downstream analyses by identifying rare, spatially-dispersed murine embryonic macrophages, predicting pleiotropic ligand-receptor interactions during limb morphogenesis, computing RNA velocity of cardiac cell fate bifurcation, and making spatially-resolved <i>in silico</i> perturbation predictions. These results demonstrate Starro's power in enabling spatial-aware single-cell level analyses for many STs, with potential to revolutionize or replace conventional scRNA-seq techniques.
</p>

## Highlights of Starro:

* <p align="justify">Starro provides the sole RNA-based, image-based, and combinatorial cell segmentation approaches, capable of analyzing all published sequencing- and imaging-based spatial transcriptomic data with subcellular resolution. </p>
* <p align="justify">Starro incorporates domain partitioning to account for the spatial heterogeneity of RNA density prior to segmentation. It optimizes the latent parameters according to local densities, segmenting cells in each partitioned domain. This process is highly integrated, working silently and automatically. </p>
* <p align="justify">Starro leverages spot-wise RNA intensity and density as two fundamental features to construct the null hypothesis, enabling highly sensitive segmentation as long as the cell is observable to the naked eye. </p>
* <p align="justify">Starro utilizes three novel algorithms, namely combinatorial Expectation Maximization and Belief Propagation (EM-BP), Modified local Moran’s I (mLMI), and Gaussian blurred-OTSU (gOTSU), to balance accuracy and data scalability in many diverse scenarios. </p>
* <p align="justify">Starro also includes spatially-resolved pleiotropic ligand-receptor interaction analysis and RNA-velocity vector field analysis, enriching downstream research possibilities. </p>


## Usage
### Installation
```
pip install git+https://github.com/Bai-Lab/Starro.git
```

### Tutorials and demo-cases
- Starro is integrated and easier to use in our comprehensive spatial transcriptomics analytic framework [Spateo](https://github.com/aristoteleo/spateo-release).
- The tutorial page for integrated Starro cell segmentation is at [Cell segmentation tutorial page](https://spateo-release.readthedocs.io/en/latest/tutorials/notebooks/cell_segmentation.html).
- A brief **tutorial** [(vignette)](https://github.com/Bai-Lab/Starro/blob/main/notebooks/starro_rna_seg_tutorial.ipynb) is also included in this standalone package.

## Reproducibility
Scripts to reproduce benchmarking and analytic results in Starro paper are in repository [Starro_scripts](https://github.com/Bai-Lab/Starro_scripts)

## Discussion 
Users can use issue tracker to report software/code related [issues](https://github.com/Bai-Lab/Starro/issues). For discussion of novel usage cases and user tips, contribution on Starro performance optimization, please contact the authors via [email](mailto:baiyinqi@genomics.cn). 
