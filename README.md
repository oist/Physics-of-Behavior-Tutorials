# Physics of Behavior Tutorials

**Overview of the tutorials**

We offer three tutorials covering the analysis of organism-scale behavior through distinct yet interconnected topics. These topics are rooted in the ideas and tools of the physics of complex systems and thus we denote this collection as "Physics of Behavior". While they are technically independent, completing them in order may be beneficial. Please cite these tutorials by using the DOI 10.5281/zenodo.15099730 and we hope to add to this collection in the future.

1. Posture Space Analysis: the first tutorial explores posture space analysis by demonstrating how to decompose high-dimensional postural data into a few meaningful eigenpostures using Principal Component Analysis (PCA). The dataset used comes from C. elegans posture tracking.
2. Reconstructing Posture Space Dynamics: the second tutorial introduces the concept of state space and chaotic systems through a toy model. It then presents a data-driven technique for reconstructing the state space of that model.
3. Markov Chains for Behavioral Modeling: the final tutorial introduces Markov chains as a framework for modeling behavior. A toy model is used to illustrate how transition matrices can naturally cluster the state space. The tutorial concludes with a real-life application using C. elegans postural data.


Applying these techniques to your own data may require additional effort. A more detailed version of state space reconstruction and Markov chain methods is available [here](https://github.com/AntonioCCosta/markov_worm/tree/main). 


**How to Use These Tutorials on Your Local Machine**

The only prerequisite for these tutorials is to have a recent version of a package manager, such as Conda or Miniforge, installed on your computer. Once that’s set up, follow these steps:

1. Download the ZIP file from GitHub to your local computer.
2. Extract the file to a convenient location.
3. Open your terminal.
4. Navigate to the extracted folder by typing:
`cd /path/to/your/folder`
5. Create the Conda environment using the provided .yml file:
`conda env create -f requirements.yml`
6. Activate the environment:
`conda activate tutorial_env`
7. Launch Jupyter Lab:
`jupyter lab`
8. Start the tutorials!


**Some Background** 

The quantitative analysis of posture-scale animal behavior is now widespread across disciplines, including computer science, neuroscience, ethology and biophysics.  As such there are many reviews and we list a few of them: [Pereira, T. D., Shaevitz, J. W., & Murthy, M. (2020). Quantifying behavior to understand the brain. Nature Neuroscience, 23, 1537–1549.](https://doi.org/10.1038/s41593-020-00734-z),[ Krakauer, J. W., Ghazanfar, A. A., Gomez-Marin, A., MacIver, M. A., & Poeppel, D. (2017). Neuroscience needs behavior: Correcting a reductionist bias. Neuron, 93(3), 480–490.](https://doi.org/10.1016/j.neuron.2016.12.041), [ Berman, G. J. (2018). Measuring behavior across scales. BMC Biology, 16, Article 23.](https://doi.org/10.1186/s12915-018-0494-7), and  [Brown, A. E. X., & de Bivort, B. (2018). Ethology as a physical science. Nature Physics, 14, 653–657.](https://doi.org/10.1038/s41567-018-0188-z).  We note that many of the conceptual ideas described in the reviews above were already present in the early posture-scale analysis of C. elegans, [Stephens, G. J., Johnson-Kerner, B., Bialek, W., & Ryu, W. S. (2008). Dimensionality and dynamics in the behavior of C. elegans. PLoS Computational Biology, 4(4), e1000028.](https://doi.org/10.1371/journal.pcbi.1000028).

The tutorials included here describe analyses for characterizing both the space of postures (eigenshapes) and the space of posture changes in time (state space and Markov chains). Posture changes over time are a direct result of muscle activations and forces between the animal and its environment. Posture changes also reflect neural control and can even convey more abstract information such as behavioral states; we are all familiar with “reading'' the body language of those around us.

In these tutorials we view behavior as a dynamical system and we incorporate insights from dynamical systems theory in order to quantitatively capture what animals do.  An excellent textbook for a basic introduction is Strogatz, S. H. (2018). Dynamical systems and chaos (2nd ed.). CRC Press. Of course, dynamical systems theory was not historically developed to understand animal behavior, and there are particular challenges associated with the modeling of living systems. These include in particular a lack of first-principles theory necessitating a data-driven approach, and the challenge of many interacting degrees of freedom, which means in principle that our descriptions should allow for high-dimensionality. Even the simplest animal functions through the expression of diverse genes in a variety of cells interconnected into complex networks!  Nevertheless, we have found that our approach offers a remarkably powerful perspective, and we take these challenges as an opportunity to develop new theoretical ideas, and thus, hopefully, a better understanding of behavior.

**Intended Audience and Required Background** 

These tutorials are designed for anyone with a general scientific background, and a college-level understanding of linear algebra. For example, we will touch on concepts such as covariance matrices and eigendecomposition without extensive introduction. Concepts related to dynamical systems—including state space, embedding, Lyapunov exponents, and Markov chains—are explained in more detail. However, some additional reading will be necessary for a deeper understanding. We highly recommend consulting the references included in the tutorials.
Many of the concepts originate from physics, but we strive to present them in a way that is accessible to anyone with a graduate-level (or advanced undergraduate) scientific background. Additionally, we have included some more advanced topics for those interested in exploring further.


**Contributions** 

These tutorials were written by Akira Kawano, Anouk Béraud and Irina Korshok under the supervision of Greg J Stephens. 

**Acknowledgments** 

We would also like to acknowledge engaging and constructive feedback from students of the JAX Short Course on the Application of Machine Learning for Automated Quantification of Behavior (2023, 2024), Cajal Course on Quantitative Approaches to Behaviour and Virtual Reality (2024), and EPFL short course on Physics of Animal Behavior (2024), as well as the useful comments of Tatsuo Izawa and Kosmas Deligkaris.

