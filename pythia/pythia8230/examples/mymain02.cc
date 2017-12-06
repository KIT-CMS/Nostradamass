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
     
//Starting Pythia

  //bool Debug = false;
     
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
     
// Fiducial cuts
//  float etamax = 2.4; //pseudorapidity must be below etamax, otherwise muons are outside detector range
//  float ptmin = 15.0; //we only accept higgs-decay-candidate muons whose pt is above ptmin.
     
// pi constant
//  float pi = 3.14159276;
  //float pi = 4.*atan(1);
         
//Creating a root file with a tree tMuon        
//  string cmndFilename = argv[1];
//  string rootFilename = cmndFilename.replace(cmndFilename.length()-4,4,"root"); 
//  TFile * f = new TFile(rootFilename.c_str(),"recreate"); //the root file containing the 
                                                          //tree now has the same name as the cmnd file that was used to create it
//Creating a root tree tHiggs
//  TTree * tHiggs = new TTree("tHiggs", "Higgs candidate muon pairs");
//  int idHiggs ;
//    float pxHiggs, pyHiggs, pzHiggs, enHiggs, massHiggs;
//    float pxPosTau, pyPosTau, pzPosTau, enPosTau;
//    float pxNegTau, pyNegTau, pzNegTau, enNegTau;
//  float cosDecayTheta;
//  float pxDimuon, pyDimuon, pzDimuon, enDimuon, massDimuon;

// tree branches ---->

//  tHiggs -> Branch("i", &eventCounter, "i/I"); // event number
//  tHiggs -> Branch("IsEventSelected", &isSelected, "IsEventSelected/O" ); // boolean variable 
//  
//  // Higgs boson kinematics
//  tHiggs -> Branch("HiggsID", &idHiggs, "HiggsID/I" );
//  tHiggs -> Branch("PxHiggs", &pxHiggs, "PxHiggs/F" );
//  tHiggs -> Branch("PyHiggs", &pyHiggs, "PyHiggs/F" );
//  tHiggs -> Branch("PzHiggs", &pzHiggs, "PzHiggs/F" );
//  
  // ******************************************
  // add new branch to store Higgs boson mass 
  // ******************************************

//  tHiggs -> Branch("PxNegMuon", &pxNegMuon, "PxNegMuon/F");
//  tHiggs -> Branch("PyNegMuon", &pyNegMuon, "PyNegMuon/F");
//  tHiggs -> Branch("PzNegMuon", &pzNegMuon, "PzNegMuon/F");
//  tHiggs -> Branch("EnNegMuon", &enNegMuon, "EnNegMuon/F");

  // ******************************************
  // add new branches to store 4-momentum 
  // of the positive muon
  // ******************************************

  // *******************************************
  // add new branches to store 4-momentum of the 
  // dimuon system.
  // -------------------------------------------
  // not obligatory :
  // calculate cosine of the Higgs decay angle 
  // (theta*) definition of this variable can be 
  // found in the exercise sheet  
  // *******************************************
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
        //std::cout << " _____________next event ___________" << iEvent << std::endl;
        // Generate event
        pythia.next();
        for (int iPart = 0; iPart < pythia.event.size(); ++iPart )
        {
			if (pythia.event[iPart].id() == 25) // Boson
				bosonIdx = iPart;
//			break;	
			// identify particle numbers for
			// Boson
			//
			// positive visible lepton
			// tau neutrino
			// anti-neutrino to visible lepton
			//
			// negative visible lepton
			// anti-tau neutrino
			// neutrino to visible lepton


 // for every particle in the event
//            if (((pythia.event[iPart].status() >0)&& (pythia.event[iPart].id() >8) && (pythia.event[iPart].id() !=21)) || pythia.event[iPart].id() ==25  || pythia.event[iPart].id() ==15 || pythia.event[iPart].id() ==-15 )

        /*    std::cout << "Particle " << iPart << "\t id: " << pythia.event[iPart].id() << "\t status: " << pythia.event[iPart].status() << 
				"\t mass: " << pythia.event[iPart].m() << 
				"\t mother1/2: " << pythia.event[iPart].mother1() << 
				"/" << pythia.event[iPart].mother2() <<
				"\t daughterList: ";
			for(size_t i = 0; i < pythia.event[iPart].daughterList().size(); ++i)
				std::cout << pythia.event[iPart].daughterList()[i] << ", ";
			std::cout<< std::endl;*/
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

// _________END_OF_PROGRAM_____________________

