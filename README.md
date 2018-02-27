# Nostradamass
## No-full sim DNN training derived approximation of mass 

Nostradamass is a full di-tau system reconstruction. It uses events generated with the Pythia 8 event generator as labled data for a deep neural network. It does not use any reconstruction-based quantities and is therefore not dependent on any experiment.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Pre-trained model files are also available for the direct application on any ROOT n-tuple following the appropriate naming scheme.

### Prerequisites

The software is optimized to run on any machine providing the sft repository in CVMFS. It has been tested with the LCG92 release, but is not bound specifically to it.

### Installing

Download the repository with

```
git clone git@github.com:rfriese/Nostradamass.git
```

Edit setup_env.sh depending on the system you use.


## Generating simulated events

Compilation requires gcc7. Earlier versions are not supported.

```
source setup_env.sh
cd pythia_eventgeneration
make
```

The job.sh script helps in setting the parameters. It can also be sent to your local batch system by the [JDL creator]("https://gitlab.ekp.kit.edu/mschnepf/jdl_creator").

The first parameter is the mass to be simulated, the second the seed, the third the channel. The last parameter is a boolean that is "true" for events where the first tau is positively charged and "false" for negatively charged first taus. It is recommended to produce always the same amount of events with both parameters.
```
./job.sh 300 1234 mt true

```
Edit eventgeneration.cc to modify its parameters.

## Performing a training

Modify the script "train_channel.sh" to point to the previously produced simulated events and provide an output path. The only command-line parameter is the channel, e.g.

```
./train_channel mt
```


## Application on a sync n-tuple

To run Nostradamass on a ROOT n-tuple, the following properties must be present:
1. pt_1, eta_1, phi_1, m_1: The four-vector of the first visible decay products of the tau, flavour-sorted
2. pt_2, eta_2, phi_2, m_2: The four-vector of the second visible decay products of the tau, flavour-sorted
3. metcov00, metcov11: The diagonal elements of the MET covariance matrix

Edit the file "config.yaml" or create a similar yaml file following your needs. The 'models' section specifies on which trees which models should be applied. List all your 'files' in the corresponding section.
```
```
