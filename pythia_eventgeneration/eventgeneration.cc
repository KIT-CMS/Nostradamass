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
#include <TFile.h>
#include <TTree.h>
#include <time.h>

using namespace Pythia8;

#define MIN_ELECTRON_PT 26
#define MAX_ELECTRON_ETA 2.6
#define MIN_MUON_PT 20
#define MAX_MUON_ETA 2.5
#define MIN_TAU_PT 30
#define MAX_TAU_ETA 2.4

void configure(char * argv[], Pythia& pythia, char flavour1, char flavour2, bool inverted, int int_mass, std::string seed)
{

    stringstream mass;
    mass << int_mass;
    stringstream massPlusOne;
    massPlusOne << (int_mass + 1);
    stringstream massMinusOne;
    massMinusOne << (int_mass - 1);

    pythia.readString("Random:seed = " + seed);

    pythia.readString("25:m0 = " + mass.str());
    pythia.readString("25:mMin = " + massMinusOne.str());
    pythia.readString("25:mMax = " + massPlusOne.str());

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HiggsSM:all = off");
    pythia.readString("HiggsSM:gg2H = on");
    pythia.readString("HiggsSM:ff2Hff(t:ZZ) = on");
    pythia.readString("HiggsSM:ff2Hff(t:WW)  = on");
    pythia.readString("Main:numberOfEvents = 10000");
    pythia.readString("Random:setSeed = true");
    pythia.readString("ProcessLevel:resonanceDecays = on");
    pythia.readString("PartonLevel:all = on");// if off, generation stops before parton level events (partons are quarks and gluons)
    pythia.readString("PartonLevel:FSR = off");//final state radiation
    pythia.readString("PartonLevel:ISR = on");// initial state radiation
    pythia.readString("PartonLevel:Remnants = off");// beam remnants
    pythia.readString("HadronLevel:all = on");//if off, generation stops before hadron level steps
    pythia.readString("HadronLevel:Hadronize = off");
    pythia.readString("HadronLevel:Decay = on");
    pythia.readString("HadronLevel:BoseEinstein = off");

    pythia.readString("PDF:pSet = 16");// specifies the parton density for proton beams

    pythia.readString("25:onMode = off");// switches off all Higgs decay channels
    pythia.readString("25:onIfMatch = 15 -15");// swiches back on higgs to tau pair


    pythia.readString("15:onMode = off");
	std::cout << flavour1 << " / " << flavour2 << std::endl;
    if ((flavour1 == 'e') && (flavour2 == 'e'))
    {
        pythia.readString("15:onIfAny = 11");
        return;
    }
    if ((flavour1 == 'm') && (flavour2 == 'm'))
    {
        pythia.readString("15:onIfAny = 13");
        return;
    }
    if ((flavour1 == 't') && (flavour2 == 't'))
    {
        pythia.readString("15:onIfAny = 211 111 321");
        return;
    }
    

    if (flavour1 == 'e')
    {
        pythia.readString("15:onPosIfAny = 11");
    }
    else if(flavour1=='m')
    {
        pythia.readString("15:onPosIfAny = 13");
    }
    else if(flavour1=='t')
    {
        pythia.readString("15:onPosIfAny = 211 111 321");
    }    


    if (flavour2 == 'e')
    {
        pythia.readString("15:onNegIfAny = 11");
    }
    else if (flavour2=='m')
    {
        pythia.readString("15:onNegIfAny = 13");
    }
    else if (flavour2=='t')
    {
        pythia.readString("15:onNegIfAny = 211 111 321");
    }

}

//typedef RMLV ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >;
class diTauEvents {
    public:
        diTauEvents(std::string out_file_name);
        ~diTauEvents();
        void next();
        void add(Particle&);
        void add(Particle*);
        void add(int&);
        void fill();
        void write();
    // particles: Boson, 2xTau, 4xNeutrino, 4xjet = 44

    private:
        static const size_t n_particles = 11;
        static const size_t n_branches = 4 * n_particles;
        float toTree[n_branches];
        int indices[n_particles+1];
        TFile* out_file;
        TTree* out_tree;

        size_t particle_index;
};

void diTauEvents::fill()
{
    out_tree->Fill();
}

void diTauEvents::write()
{
    out_tree->Write();
}

void diTauEvents::add(Particle &in)
{
    size_t i = particle_index * 4;
    toTree[i++] = in.e();
    toTree[i++] = in.px();
    toTree[i++] = in.py();
    toTree[i++] = in.pz();
    indices[particle_index++] = in.id();
}

void diTauEvents::add(Particle *in)
{
    size_t i = particle_index * 4;
    toTree[i++] = in->e();
    toTree[i++] = in->px();
    toTree[i++] = in->py();
    toTree[i++] = in->pz();
    indices[particle_index++] = in->id();
}

void diTauEvents::add(int& in)
{
    indices[particle_index++] = in;
}

void diTauEvents::next()
{
    particle_index = 0;
    for(size_t i = 0; i<n_branches; ++i)
    {
        toTree[i] = (float) i;
    }
}

diTauEvents::~diTauEvents()
{
    out_file->Close();
}

diTauEvents::diTauEvents(std::string out_file_name)
{
    out_file = new TFile(out_file_name.c_str(), "RECREATE");
    out_tree = new TTree("tree", "tree");

    std::string particle_postfix[] = {"B", "1", "2", "t1n", "l1n", "t2n", "l2n", "j1", "j2", "j3", "j4"};
    std::string particle_prefix[] = {"e", "px", "py", "pz"};
    size_t i = 0;
    size_t j = 0;
    for(const string& p: particle_postfix)
    {
        out_tree->Branch(("id_"+p).c_str(), &indices[j++], ("id_"+p+"/I").c_str());
        for(const string& q: particle_prefix)
            {
                out_tree->Branch((q+"_"+p).c_str(), &toTree[i++], (q+"_"+p+"/F").c_str());
            }
    } 
	out_tree->Branch("production_mode", &indices[j++], "production_mode/I");

}

Particle* noParticle()
{
    Vec4 empyt_vec(0,0,0,0);
    return new Particle (0, 0, 0, 0, 0, 0, 0, 0, empyt_vec);//, double mIn=0., double scaleIn=0., double polIn=9.)
}

Particle* sumParticles(const Event &e, const std::vector<int> &in)
{
    Vec4 vec(0,0,0,0);
    for(auto i: in)
    {
        vec += e[i].p();
    }

    return new Particle (0, 0, 0, 0, 0, 0, 0, 0, vec);//, double mIn=0., double scaleIn=0., double polIn=9.)
}

bool fiducialCuts(Particle* p)
{
    if (abs(p->id()) == 11)
        return ((p->pT() > (MIN_ELECTRON_PT)) && abs(p->eta()) < (MAX_ELECTRON_ETA));
    else if (abs(p->id()) == 13)
        return ((p->pT() > (MIN_MUON_PT)) && abs(p->eta()) < (MAX_MUON_ETA));
    else
        return ((p->pT() > (MIN_TAU_PT)) && abs(p->eta()) < (MAX_TAU_ETA));
}


std::vector<int> upToSeveralMothers(Pythia &pythia, int index)
{
    if(pythia.event[index].motherList().size()==1)
        return upToSeveralMothers(pythia, pythia.event[index].mother1());
    else
        return pythia.event[index].motherList();
}

int cousin(Pythia &pythia, int index, int index_previous)
{
    if(pythia.event[index].motherList().size() == 0)
        return 0;
    if(pythia.event[index].daughterList().size()==1)
        return cousin(pythia, pythia.event[index].mother1(), index);
    else
    {
        auto daughters = pythia.event[index].daughterList();
        for(auto d: daughters)
        {
            if(d!= index_previous)
                return d;
        }
    }
	return 0;
}

int downToEnd(Pythia &pythia, int index)
{
    if( pythia.event[index].isFinal())
        return index;
    else
	{
		if (pythia.event[index].daughterList().size()>1)
			return 0;
        return downToEnd(pythia, pythia.event[index].daughter1());
	}
}

std::vector<int> getJet_indices(Pythia &pythia, int i_lep, int genBosonIdx)
{
    auto mothers = pythia.event[genBosonIdx].motherList();
    auto bosonParents = upToSeveralMothers(pythia, genBosonIdx);
    std::vector<int> i_finalStateJets;
    for(auto parent: bosonParents)
    {
        auto p = parent;
        int c;
		for(size_t a=1; a<pythia.event[p].daughterList().size(); a++)
		{
        	i_finalStateJets.push_back(pythia.event[p].daughterList()[a]);
		}
        c = cousin(pythia, pythia.event[p].mother1(), p);
        if(c!=0)
            i_finalStateJets.push_back(c);
		
        do
        {
            c = cousin(pythia, pythia.event[pythia.event[p].mother1()].mother1(), pythia.event[p].mother1());
            p = c;
            if(c!=0)
                i_finalStateJets.push_back(c);
        }
        while ( c != 0 );
    }
	// remove duplicates
	sort( i_finalStateJets.begin(), i_finalStateJets.end() );
    i_finalStateJets.erase( unique( i_finalStateJets.begin(), i_finalStateJets.end() ), i_finalStateJets.end() );
    for(auto j: i_finalStateJets)
	{
		j = downToEnd(pythia, j);
	}
	// sort by pT
    std::sort(std::begin(i_finalStateJets ), std::end(i_finalStateJets ), [&pythia](int a, int b) {return pythia.event[a].pT() > pythia.event[b].pT(); });
	// eta cut
	i_finalStateJets.erase(std::remove_if(i_finalStateJets.begin(), i_finalStateJets.end(), 
                       [&pythia](int i) { return (std::abs(pythia.event[i].eta()) > 4.7); }), i_finalStateJets.end());
    return i_finalStateJets;
}

int main( int argc, char * argv[] )
{

    clock_t clkStart;
    clock_t clkFinish;
    clkStart = clock();
    assert(argc == 5);

    Pythia pythia;

    istringstream in_mass(argv[1]);
    stringstream in_seed(argv[2]);
    stringstream in_channel(argv[3]);
    std::string channel = in_channel.str();

    stringstream in_invert(argv[4]);
    std::string str_invert= in_invert.str();
    bool inverted = (str_invert != "false");

    char flavour1, flavour2;
    
    flavour1 = channel[inverted];
    flavour2 = channel[!inverted];
     
    int int_mass;
    in_mass >> int_mass;
    configure(argv, pythia, flavour1, flavour2, inverted, int_mass, in_seed.str());

    diTauEvents evt("m_" + in_mass.str() + "_" + in_seed.str() + "_" +channel+"_"+str_invert+".root");

    // actually run the event generation 
    pythia.init();  // pythia initialization   
    pythia.settings.listChanged();
    pythia.particleData.listChanged();
    int nEvents = pythia.mode("Main:numberOfEvents"); //the .cmnd file tells us of the value of Main:numberofEvents;
     
    int negTauIdx, posTauIdx;
    int negTauNeutrinoIdx, posTauNeutrinoIdx;

    Particle *posLepNeutrino, *posVisible;
    Particle *negLepNeutrino, *negVisible;
	Particle *jet1, *jet2, *jet3, *jet4;

    int bosonIdx = 0;
    int genBosonIdx = 0;
    std::vector<int> negTauDaughters, posTauDaughters;
    std::vector<int> negTauVis, posTauVis;
    int tries = 0;
    bool writeOutput = true;
	int production_mode;
    for ( int iEvent = 0; iEvent < nEvents; ++iEvent )
    {
        if (( tries%100 == 0) && writeOutput)
        {
            clkFinish = clock();
            std::cout << "Events/tries: " << iEvent << " / " << tries << ", runtime (s): " << (clkFinish - clkStart)/1000000 << std::endl;
            writeOutput = false;
        }
        if ( tries%100 == 1)
            writeOutput = true;
        ++tries;
        negTauDaughters.clear();
        posTauDaughters.clear();
        negTauVis.clear();
        posTauVis.clear();
        pythia.next();
//        std::cout << "\n Next Event "<< std::endl;
        // find the index if the first generated boson: genBosonIdx
        // find the index of the boson decaying to the two taus : bosonIdx
        for (int iPart = 0; iPart < pythia.event.size(); ++iPart )
        {
            if (pythia.event[iPart].id() == 25 && pythia.event[iPart].status() == -22) // Boson
                genBosonIdx = iPart;
            if (pythia.event[iPart].id() == 25) // Boson
                bosonIdx = iPart;
        }

        //positive lepton
        posTauIdx = pythia.event[bosonIdx].daughter1();
        posTauNeutrinoIdx = pythia.event[posTauIdx].daughter1();
        negTauIdx = pythia.event[bosonIdx].daughter2();
        negTauNeutrinoIdx = pythia.event[negTauIdx].daughter1();

        bool goodEvent = true; 


        posTauDaughters = pythia.event[posTauIdx].daughterList();
//        std::cout << "daughters: " << std::endl;
//        for(auto i: posTauDaughters)
//            std::cout << pythia.event[i].id() << ", ";
        if(pythia.event[posTauDaughters[1]].id() == 11 || pythia.event[posTauDaughters[1]].id() == 13)
        {
            posLepNeutrino = new Particle(pythia.event[posTauDaughters[2]]);
            posVisible = new Particle(pythia.event[posTauDaughters[1]]);
        }
        else
        {
            posLepNeutrino = noParticle();
            posTauDaughters.erase(posTauDaughters.begin());
            posVisible = sumParticles(pythia.event, posTauDaughters);
        }
        negTauDaughters = pythia.event[negTauIdx].daughterList();
        if(pythia.event[negTauDaughters[1]].id() == -11 || pythia.event[negTauDaughters[1]].id() == -13)
        {
            negLepNeutrino = new Particle(pythia.event[negTauDaughters[2]]);
            negVisible = new Particle(pythia.event[negTauDaughters[1]]);
        }
        else
        {
            negLepNeutrino = noParticle();
            negTauDaughters.erase(negTauDaughters.begin());
            negVisible = sumParticles(pythia.event, negTauDaughters);
        }

        goodEvent = goodEvent && fiducialCuts(posVisible);
        goodEvent = goodEvent && fiducialCuts(negVisible);

        if (!goodEvent)
        {
            --iEvent;
            continue;
        }
        // get the indices of the final state jets
        auto posTauGluonJet = getJet_indices(pythia, 0, genBosonIdx);
		jet1 = posTauGluonJet.size() > 0 ? (new Particle(pythia.event[posTauGluonJet[0]])) : noParticle();
		jet2 = posTauGluonJet.size() > 1 ? (new Particle(pythia.event[posTauGluonJet[1]])) : noParticle();
		jet3 = posTauGluonJet.size() > 2 ? (new Particle(pythia.event[posTauGluonJet[2]])) : noParticle();
		jet4 = posTauGluonJet.size() > 3 ? (new Particle(pythia.event[posTauGluonJet[3]])) : noParticle();

		if((pythia.event[3].daughterList().size() == 3) && (pythia.event[4].daughterList().size() == 3))
			production_mode = 2;
		else
			production_mode = 1;
        // writing out
        evt.next();
        evt.add(pythia.event[bosonIdx]);
/*        for (int iPart = 0; iPart < pythia.event.size(); ++iPart )
        {
           std::cout << iPart << " :\t id:" << pythia.event[iPart].id() << ", pt: " << pythia.event[iPart].pT() << ";" << pythia.event[iPart].m() << ", \t, daughters:";
           for(auto d:  pythia.event[iPart].daughterList())
               std::cout << d << ",";
            std::cout << " \t; mothers:";

           for(auto d: pythia.event[iPart].motherList())
                std::cout << d << ",";
            std::cout << "status/final: " << pythia.event[iPart].status() << "/" << pythia.event[iPart].isFinal() << std::endl;
        }
        std::cout << "final state particles: " << std::endl;
        for (int iPart = 0; iPart < pythia.event.size(); ++iPart )
        {
            if(!pythia.event[iPart].isFinal() || pythia.event[iPart].status()==91)
                continue;
           std::cout << iPart << " :\t id:" << pythia.event[iPart].id() << ", pt: " << pythia.event[iPart].pT() << ", \t, daughters:";
           for(auto d:  pythia.event[iPart].daughterList())
               std::cout << d << ",";
            std::cout << " \t; mothers:";

           for(auto d: pythia.event[iPart].motherList())
                std::cout << d << ",";
            std::cout << "status/final: " << pythia.event[iPart].status() << "/" << pythia.event[iPart].isFinal() << std::endl;
        }*/

        if(!inverted)
        {
            evt.add(posVisible);
            evt.add(negVisible);
            evt.add(pythia.event[posTauNeutrinoIdx]);
            evt.add(posLepNeutrino);
            evt.add(pythia.event[negTauNeutrinoIdx]);
            evt.add(negLepNeutrino);
        }
        else
        {
            evt.add(negVisible);
            evt.add(posVisible);
            evt.add(pythia.event[negTauNeutrinoIdx]);
            evt.add(negLepNeutrino);
            evt.add(pythia.event[posTauNeutrinoIdx]);
            evt.add(posLepNeutrino);
        }
		evt.add(jet1);
		evt.add(jet2);
		evt.add(jet3);
		evt.add(jet4);
		evt.add(production_mode);
        delete posLepNeutrino;
        delete posVisible;
        delete negLepNeutrino;
        delete negVisible;
		delete jet1;
		delete jet2;
		delete jet3;
		delete jet4;

        evt.fill();
    }
    evt.write();
    std::cout << "Summary Events/tries: " << nEvents << " / " << tries << ", runtime: " << clkFinish - clkStart << std::endl;
    evt.~diTauEvents();
  return 0;

}
