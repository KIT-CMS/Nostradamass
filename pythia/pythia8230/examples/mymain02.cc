// _________________________________________________________
//
// Program illustrating generation of events with Pythia8 
// and storage of kinematics in TTree 
// _________________________________________________________

#include <assert.h>
#include <string>
#include <iostream>
#include <cmath>
#include "Pythia8/Pythia.h"
//#include "TFile.h" 
//#include "TH1.h" 
//#include "TTree.h"
//#include "TROOT.h"
//#include "TLorentzVector.h"

using namespace Pythia8;

void write_particle(Particle &in)
{
	std::cout << in.e() << "," << in.px() << "," << in.py() << "," << in.pz() << "," << in.id() << ";";
}

void write_particle(int &in, Event &e)
{
	if (in > 0)
		write_particle(e[in]);
	std::cout << "|";
}

void write_particles(std::vector<int> &in, Event &e)
{
	for(size_t i = 0; i < in.size(); ++i)
	{
		write_particle(e[in[i]]);
	}
	std::cout << "|";
}

int main( int argc, char * argv[] )
{
	assert(argc == 4);
    Pythia pythia;

    stringstream in_seed(argv[3]);
    pythia.readString("Random:seed = " + in_seed.str());
     
    pythia.readFile(argv[1]); // looks at the .cmnd file for settings
    istringstream in_mass(argv[2]);
    int int_mass;
    in_mass >> int_mass;
    stringstream mass;
    mass << int_mass;
    stringstream massPlusOne;
    massPlusOne << (int_mass + 1);
    stringstream massMinusOne;
    massMinusOne << (int_mass - 1);
    pythia.readString("25:m0 = " + mass.str());
    pythia.readString("25:mMin = " + massMinusOne.str());
    pythia.readString("25:mMax = " + massPlusOne.str());
     
    pythia.init();  // pythia initialization   
    pythia.settings.listChanged(); // pythia status
    pythia.particleData.listChanged(); // pythia status
    int nEvents = pythia.mode("Main:numberOfEvents"); //the .cmnd file tells us of the value of Main:numberofEvents;
     
	int negTauIdx, posTauIdx;
	int negTauNeutrinoIdx, posTauNeutrinoIdx;
	int negLepNeutrinoIdx, posLepNeutrinoIdx;
	int bosonIdx = 0;
	std::vector<int> negTauDaughters, posTauDaughters;
	std::vector<int> negTauVis, posTauVis;
    for ( int iEvent = 0; iEvent < nEvents; ++iEvent )
    {
		negTauDaughters.clear();
		posTauDaughters.clear();
		negTauVis.clear();
		posTauVis.clear();
        pythia.next();
        for (int iPart = 0; iPart < pythia.event.size(); ++iPart )
        {
			if (pythia.event[iPart].id() == 25) // Boson
				bosonIdx = iPart;
        }

		//positive lepton
		posTauIdx = pythia.event[bosonIdx].daughter1();
		posTauNeutrinoIdx = pythia.event[posTauIdx].daughter1();
		assert(pythia.event[posTauNeutrinoIdx].id() == 16);

		posTauDaughters = pythia.event[posTauIdx].daughterList();
		// leptonic decay
		if(pythia.event[pythia.event[posTauIdx].daughter2()].id() == -12 || pythia.event[pythia.event[posTauIdx].daughter2()].id() == -14)
		{
			posLepNeutrinoIdx = posTauDaughters[2];
			posTauVis.push_back(posTauDaughters[1]);
		}
		else
		{
			posLepNeutrinoIdx = 0;
			posTauDaughters.erase(posTauDaughters.begin());
			posTauVis = posTauDaughters;
		}

		//std::cout << "bosonid: " << bosonIdx << ", posTau: " << posTauIdx << ", posTauNeutrino: " << posTauNeutrinoIdx << ", posLepNeutrinoIdx: " << posLepNeutrinoIdx << ", posTauVis: ";
		//for(size_t i = 0; i < posTauVis.size(); ++i)
		//	std::cout << posTauVis[i] << "/";
		//std::cout << std::endl;

		//negative lepton
		negTauIdx = pythia.event[bosonIdx].daughter2();
		negTauNeutrinoIdx = pythia.event[negTauIdx].daughter1();
		assert(pythia.event[negTauNeutrinoIdx].id() == -16);

		negTauDaughters = pythia.event[negTauIdx].daughterList();
		// leptonic decay
		if(pythia.event[pythia.event[negTauIdx].daughter2()].id() == 12 || pythia.event[pythia.event[negTauIdx].daughter2()].id() == 14)
		{
			negLepNeutrinoIdx = pythia.event[negTauIdx].daughter2();
			negTauVis.push_back(negTauDaughters[1]);
		}
		else
		{
			negLepNeutrinoIdx = 0;
			negTauDaughters.erase(negTauDaughters.begin());
			negTauVis = negTauDaughters;
		}

		//std::cout << "bosonid: " << bosonIdx << ", negTau: " << negTauIdx << ", negTauNeutrino: " << negTauNeutrinoIdx << ", negLepNeutrinoIdx: " << negLepNeutrinoIdx << ", negTauVis: ";
		//(for(size_t i = 0; i < negTauVis.size(); ++i)
		//	std::cout << negTauVis[i] << "/";
		//std::cout << std::endl;
		
		bool goodEvent = true;
		goodEvent == goodEvent && pythia.event[posTauIdx].pT() > 15 && abs(pythia.event[posTauIdx].eta()) < 2.7;
		goodEvent == goodEvent && pythia.event[negTauIdx].pT() > 15 && abs(pythia.event[negTauIdx].eta()) < 2.7;


		if (!goodEvent)
		{
			--iEvent;
			continue;
		}

		std::cout << pythia.event[bosonIdx].m() << "|";

		std::vector<int> particles_vector;
		for(size_t i = 0; i < posTauVis.size(); ++i)
			particles_vector.push_back(posTauVis[i]);
		particles_vector.push_back(posTauNeutrinoIdx);
		particles_vector.push_back(posLepNeutrinoIdx);
		for(size_t i = 0; i < negTauVis.size(); ++i)
			particles_vector.push_back(negTauVis[i]);
		particles_vector.push_back(negTauNeutrinoIdx);
		particles_vector.push_back(negLepNeutrinoIdx);
		Particle tmp;
		//for(size_t i = 0; i < particles_vector.size(); ++i)
		//{
		//	Particle p = pythia.event[i];
		//	tmp = tmp + p;
		//}


		write_particles(posTauVis, pythia.event);
		write_particle(posTauNeutrinoIdx, pythia.event);
		write_particle(posLepNeutrinoIdx, pythia.event);

		write_particles(negTauVis, pythia.event);
		write_particle(negTauNeutrinoIdx, pythia.event);
		write_particle(negLepNeutrinoIdx, pythia.event);
		std::cout << std::endl;

  }     
  //pythia.statistics();
  return 0;

}
