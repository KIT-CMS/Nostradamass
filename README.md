# Nostradamass
## Neutrino kinematics estimation by regression with artificial intelligence to reconstruct the di-tau mass

Nostradamass is a full di-tau system reconstruction. It uses events generated with the Pythia 8 event generator as labled data for the training of a deep neural network. It does not use any reconstruction-based quantities and is therefore not dependent on any experiment.

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

Edit setup_env.sh depending on the system you use and setup_training.sh.

## Application on a sync n-tuple

To run Nostradamass on a ROOT n-tuple, the following properties must be present:
1. pt_1, eta_1, phi_1, m_1: The four-vector of the first visible decay products of the tau, flavour-sorted
2. pt_2, eta_2, phi_2, m_2: The four-vector of the second visible decay products of the tau, flavour-sorted
3. metcov00, metcov11: The diagonal elements of the MET covariance matrix

Edit the file "config.yaml" or create a similar yaml file following your needs. The 'models' section specifies on which trees which models should be applied. List all your 'files' in the corresponding section. Run the application with:
```
source setup_env.py
python calculate_mass.py config.yaml
```

The output folder will contain for each input root-file a new file containing a Tree that is a friend of the original Tree. Use the TTree::AddFriend() function to include the Nostradamass results to your original tree.


## Generating simulated events

This step is only necessary to produce new trainings. Skip this step in case you want to use existing trainings.

Compilation requires gcc62. Earlier versions are not supported. Experience shows that only slc6 machines compile properly.

```
source setup_env.sh
cd pythia_eventgeneration
make
```

The job.sh script helps in setting the parameters. It can also be sent to your local batch system by the [JDL creator]("https://gitlab.ekp.kit.edu/mschnepf/jdl_creator").

The first parameter is the mass to be simulated, the second the seed, the third the channel. The last parameter is a boolean that is "true" for events where the first tau is positively charged and "false" for negatively charged first taus. It is recommended to produce always the same amount of events with both parameters.

Make sure you run the 'job.sh' script in a new shell. If e.g. a LCG release has been sourced before, it will most likely fail!
```
./job.sh 300 1234 mt true

```
Edit eventgeneration.cc to modify its parameters like kinematic selection, number of events etc.

## Performing a training

Modify the script "train_channel.sh" to point to the previously produced simulated events and provide an output path. The first parameter is the channel, the second one the GPU number for the training.

```
source setup_training.sh
./train_Nostradamass.sh mt 0
```

