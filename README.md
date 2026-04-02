# Overview of the Emulators <a name="overview"></a>

Emulators are Machine Learning models that map cosmological (and nuisance) parameters to cosmological data vectors, just like many numerical packages for corresponding observables. The advantage of emulators is the fast evaluation time. As cosmology moves into an era in which increasingly precise and high-resolution experiments require more and more computational resources, many traditional numerical packages become too slow in terms of evaluation time. Emulators are developed to overcome this issue and can widely replace many numerical packages in MCMCs.

In this repository, we provide emulators for various data vectors and in different architectures. The architectures include Residual Multi-Layer Perceptron, Transformer, Convolutional Neural Network, and Gaussian processing models.

## CMB Emulators <a name="CMB"></a>

Cosmic Microwave Background (CMB) emulators are trained to emulate `TT` (temperature auto spectrum), `TE` (temperature-polarization cross spectrum), and `EE` (polarization auto spectrum) power spectra output calculated by the `CAMB` Boltzmann code. 

We provide a few architecture options; for instance, the users can select `modeltype` to be either `TRF` (standing for Transformer) or `CNN` (standing for Convolutional Neural Network).  

Users then need to specify the corresponding emulator, which corresponds to a specific cosmological model, as well as a particular CAMB version and accuracy settings. This specification is done as follows.

**Step :one:**: Select the 'ordering' list, which needs to be the exact sequence of parameters input to the emulator.,

**Step :two:**: Select the emulator information files as shown below

    #Switch the prefix xy for (tt, te, ee)
    'xyfilename':   emulator model parameters and 
    'xyextraname': normalization factors, . 

**Step :three:**: Specify `ellmax`, which is the largest ell mode that the emulator is trained to evaluate.

**Step :four:**: If users want to sample the parameter `theta_*` (recommended) instead of `H_0`, specify the following parameters

     'thetaordering': # exact sequence of parameters input to the Gaussian Processing emulator.
     'GPfilename':    # Gaussian Processing for the mapping between theta_* and thetaordering params to H_0.
     'GPextraname':   # Gaussian Processing for the mapping between theta_* and thetaordering params to H_0.
     
## SN and BAO Emulators <a name="SNBAO"></a>

For Supernovae luminosity distances and BAO observable emulators, we adopt ResMLP for luminosity distance and Hubble parameter. On top of that, we have a Gaussian processing emulator for $r_{drag}$. The users again need to specify the emulator parameter, normalization factor file, and the corresponding PCA transformation matrix file directories. The specific redshift z grid file is also required. Furthermore, we have a parameter named 'offset' which refers to a possible artificial shift to the dataset for distance emulators, which is due to the fact that we need to lift all data points above 0 when there are negative distances for negative redshift during training (for the sake of better interpolation). Thus, we can take the logarithm of the distance to better emulate this function.

**Step :one:**: Select the 'ordering' list, which needs to be the exact sequence of parameters input to the emulator.,

**Step :two:**: Select the Hubbe Parameter emulator information files as shown below

    'filename':   emulator model parameters and 
    'extraname': normalization factors, . 

**Step :three:**: Specify `z` arrays for both H and distance, which are the redshift arrays on which the emulator will compute the data vector. Internally, we then save an interpolator to output values of H and distance at any redshift within the range.

**Step :four:**: Internally integrate to get the supernovae distance based on the output of Hubble emulator

**Step :five:**: Similarly specify the 'filename' and 'extraname' for the GP for $r_{drag}$ for BAO likelihoods.

# Credits <a name="appendix_proper_credits"></a>

<p align="center">
<img width="780" alt="students" src="https://github.com/user-attachments/assets/fbb00dc8-c196-43cc-9bb8-cea0bd929d5e">
</p>

This repo exists only because of the hard work of the Ph.D. students [Victoria Lloyd](https://vivianmiranda.academic.ws/people/3369-victoria-lloyd), [Evan Saraivanov](https://vivianmiranda.academic.ws/people/3365-evan-saraivanov), and [Yijie Zhu](https://vivianmiranda.academic.ws/people/3366-yijie-zhu), and was developed after the science of emulators was painstakingly tested in the following papers.

    @article{Zhu:2025jim,
        author = "Zhu, Yijie and Saraivanov, Evan and Kable, Joshua A. and Giannakopoulou, Artemis Sofia and Nijjar, Amritpal and Miranda, Vivian and Bonici, Marco and Eifler, Tim and Krause, Elisabeth",
        title = "{Attention-based Neural Network Emulators for Multi-Probe Data Vectors Part III: Modeling The Next Generation Surveys}",
        eprint = "2505.22574",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        month = "5",
        year = "2025"
    }

    @article{Saraivanov:2024soy,
        author = "Saraivanov, Evan and Zhong, Kunhao and Miranda, Vivian and Boruah, Supranta S. and Eifler, Tim and Krause, Elisabeth",
        title = "{Attention-based neural network emulators for multiprobe data vectors. II. Assessing tension metrics}",
        eprint = "2403.12337",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        doi = "10.1103/PhysRevD.111.123520",
        journal = "Phys. Rev. D",
        volume = "111",
        number = "12",
        pages = "123520",
        year = "2025"
    }

    @article{Zhong:2024xuk,
        author = "Zhong, Kunhao and Saraivanov, Evan and Caputi, James and Miranda, Vivian and Boruah, Supranta S. and Eifler, Tim and Krause, Elisabeth",
        title = "{Attention-based neural network emulators for multiprobe data vectors. I. Forecasting the growth-geometry split}",
        eprint = "2402.17716",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        doi = "10.1103/PhysRevD.111.123519",
        journal = "Phys. Rev. D",
        volume = "111",
        number = "12",
        pages = "123519",
        year = "2025"
    }

The original code we used in these three papers has only partial integration to Cocoa and was based on an underlying infrastructure developed by Supranta S. Boruah for the paper (please cite it as well)

    @article{Boruah:2022uac,
        author = "Boruah, Supranta S. and Eifler, Tim and Miranda, Vivian and M, Sai Krishanth P.",
        title = "{Accelerating cosmological inference with Gaussian processes and neural networks {\textendash} an application to LSST Y1 weak lensing and galaxy clustering}",
        eprint = "2203.06124",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        doi = "10.1093/mnras/stac3417",
        journal = "Mon. Not. Roy. Astron. Soc.",
        volume = "518",
        number = "4",
        pages = "4818--4831",
        year = "2022"
    }

Our Matter Power Spectrum emulator also depends on a modified version of `symbolic_pk`. We basically use the EH code with modifications we are implementing to make it faster. So please also cite

    @article{Bartlett:2023cyr,
        author = "Bartlett, Deaglan J. and Kammerer, Lukas and Kronberger, Gabriel and Desmond, Harry and Ferreira, Pedro G. and Wandelt, Benjamin D. and Burlacu, Bogdan and Alonso, David and Zennaro, Matteo",
        title = "{A precise symbolic emulator of the linear matter power spectrum}",
        eprint = "2311.15865",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        doi = "10.1051/0004-6361/202348811",
        journal = "Astron. Astrophys.",
        volume = "686",
        pages = "A209",
        year = "2024"
    }
