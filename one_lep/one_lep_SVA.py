#!/usr/bin/env python2.7


import ROOT
import numpy as np
import pandas as pd

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.Macro( '$ROOTCOREDIR/scripts/load_packages.C' )

# Initialize the xAOD infrastructure:
if(not ROOT.xAOD.Init().isSuccess()): print "Failed xAOD.Init()"


#################################
########### Functions ###########
#################################

def find_child(p):
	#takes a particles container and retruns a list of the child particle container
	for i in range(p.nChildren()):
		if p.child(i).pdgId() != p.pdgId():
			return [p.child(i) for i in range(p.nChildren())]
		else:
			return find_child(p.child())

def find_parent(p):
	#takes a particles container and retruns a list of the child particle container
	for i in range(p.nParents()):
		if p.parent(i).pdgId() != p.pdgId():
			return [p.child(i) for i in range(p.nParents())]
		else:
			return find_parent(p.parent())

def pdg(n):
	#takes a particles container and returns the pdgId
	return n.pdgId()

def PT(n):
	#takes a particles container and returns the pdgId
	return n.p4().Pt()



def bubblesort(list):
#sort particle list accourding to their pt
# Swap the elements to arrange in order
    for iter_num in range(len(list)-1,0,-1):
        for idx in range(iter_num):
            if list[idx].p4().Pt()<list[idx+1].p4().Pt():
                temp = list[idx]
                list[idx] = list[idx+1]
                list[idx+1] = temp



#################################
######### Opening File ##########
#################################


files_dir = "/cluster/home/jcardenas34/AthDerivation_21.2.6.0/source/"

hplus_dir = "/cluster/home/jcardenas34/AthDerivation_21.2.6.0/run/hplus_1lep/"

hplus_1lep = files_dir+"DAOD_TRUTH1_1lep.test.pool.root"

one_lep_mass_350 = files_dir+"DAOD_TRUTH1._Hplus_mass_350.root"
one_lep_mass_400 = files_dir+"DAOD_TRUTH1._Hplus_mass_400.root"
one_lep_mass_500 = files_dir+"DAOD_TRUTH1._Hplus_mass_500.root"
one_lep_mass_600 = files_dir+"DAOD_TRUTH1._Hplus_mass_600.root"
one_lep_mass_700 = files_dir+"DAOD_TRUTH1._Hplus_mass_700.root"
one_lep_mass_800 = files_dir+"DAOD_TRUTH1._Hplus_mass_800.root"
one_lep_mass_900 = files_dir+"DAOD_TRUTH1._Hplus_mass_900.root"
one_lep_mass_1000 = files_dir+"DAOD_TRUTH1._Hplus_mass_1000.root"
one_lep_mass_1200 = files_dir+"DAOD_TRUTH1._Hplus_mass_1200.root"
one_lep_mass_1400 = files_dir+"DAOD_TRUTH1._Hplus_mass_1400.root"
one_lep_mass_1600 = files_dir+"DAOD_TRUTH1._Hplus_mass_1600.root"
ttbar = files_dir+"DAOD_TRUTH1_ttbar.13470208._000496.pool.root.1"




file_chain = [one_lep_mass_350, one_lep_mass_400, one_lep_mass_500, one_lep_mass_600 , one_lep_mass_700, one_lep_mass_800, one_lep_mass_900, one_lep_mass_1000, one_lep_mass_1200, one_lep_mass_1400, one_lep_mass_1600, ttbar]
file_name = ["mass_350", "mass_400", "mass_500", "mass_600" , "mass_700", "mass_800", "mass_900", "mass_1000", "mass_1200", "mass_1400", "mass_1600", "ttbar"]

#new

dR_c1_c0 = ROOT.TH1F("dR_chi1_chi0", " deltaR between chi1 and chi 0[1] ", 30 , -10,10)
	
dR_l1_c0 = ROOT.TH1F("dR_l1_chi0", " deltaR between lepton and chi 0[2] ", 30 , -6,6)
	


########################################
############# Code Starts ##############
########################################

#new change [10] to k

for k in range(len(file_chain)):
	#f = ROOT.TFile.Open(file_chain[11], "READONLY")
	f = ROOT.TFile.Open("/cluster/home/bataju/madgraph/MG5_aMC_v2_6_4/mssm_1lep/Events/run_02/DAOD_TRUTH1.test.pool.root", "READONLY")
	t = ROOT.xAOD.MakeTransientTree(f, "CollectionTree")  # makes all the objects XAOD
	  

	##========================
	print "Number of input events:", t.GetEntries()
	print "Working on ", file_chain[k]

	lepton_pt_l = []
	num_jet_l = []
	pt_leading_jets_l= []
	pt_second_jets_l=[]
	Met1_l=[]
	Met0_l =[]
	MT_l=[]
	Met_0 =[]
	ratio_leading_lep =[]
	dR_lep_jet = []
	eta = []
	phi = [] 
	eta_jet = []
	phi_jet = []
	phi_met = []
	eta_jet_s = []
	phi_jet_s = []
	k=11

	for entry in xrange(t.GetEntries()):
		t.GetEntry(entry)
		taus = t.TruthTaus
		bsm = t.TruthBSM
		neturino = t.TruthNeutrinos
		Met = t.MET_Truth
		jets = t.AntiKt4TruthDressedWZJets
		top = t.TruthTop
		bosons = t.TruthBoson
		muons = t.TruthMuons
		electrons = t.TruthElectrons
		all_particles = t.TruthParticles
		
		lepton_vector = ROOT.TLorentzVector(0,0,0,0)
		electron_vector = ROOT.TLorentzVector()
		muons_vector = ROOT.TLorentzVector()


		#=========================
		print "Number of input events:", t.GetEntries()
		print "Working on ", file_chain[k]
		print entry*100/t.GetEntries(), "% complete."
		print "This is entry number: ", entry+1
		#=========================
		# print Met.get(1).met()
		metvector0 = ROOT.TLorentzVector(0,0,0,0)
  		metvector0.SetPtEtaPhiM(Met.get(0).met(), 0, Met.get(0).phi(), 0)

  		metvector1 = ROOT.TLorentzVector(0,0,0,0)
  		metvector1.SetPtEtaPhiM(Met.get(1).met(), 0, Met.get(1).phi(), 0)

#####################################
############### BKG #################
#####################################
		print k
		if k ==11:
			electon_list = [p for p in electrons if len(electrons)>1]
			muon_list = [p for p in muons if len(muons)>=1 ]
			jet_list = [p for p in jets if len(jets)>=1 ]
			lepton_list = electon_list + muon_list

			if len(lepton_list) ==0:
				continue
			# bubblesort(electon_list)
			# bubblesort(muon_list)
			# bubblesort(jet_list)
			# bubblesort(lepton_list)
			electon_list.sort(reverse = True, key=(lambda l: l.p4().Pt()))
			muon_list.sort(reverse = True, key=(lambda l: l.p4().Pt()))
			jet_list.sort(reverse = True, key=(lambda l: l.p4().Pt()))
			lepton_list.sort(reverse = True, key=(lambda l: l.p4().Pt()))

			el_list=map(PT,electon_list)
			mu_list=map(PT,muon_list)
			jets_list=map(PT,jet_list)
			lepton_list_pt= map(PT,lepton_list)
			
			if len(lepton_list) ==0: continue
			if len(jets) <2: continue


			MT = (lepton_list[0].p4()+metvector0).Mt()
			
			eta.append(lepton_list[0].eta())
			phi.append(lepton_list[0].phi())
			eta_jet.append(jets[0].eta())
			phi_jet.append(jets[0].phi())
			eta_jet_s.append(jets[1].eta())
			phi_jet_s.append(jets[1].phi())
	
			phi_met.append(Met.get(0).phi())

			lepton_pt_l.append(lepton_list[0].p4().Pt())
			num_jet_l.append(len(jets))
			pt_leading_jets_l.append(jets[0].p4().Pt())
			pt_second_jets_l.append(jets[1].p4().Pt())
			Met1_l.append(Met.get(1).met())
			Met0_l.append(Met.get(0).met())
			dR_lep_jet.append(lepton_list[0].p4().DeltaR(jets[0].p4()))
			MT_l.append(MT)

########################################
############# Signal ###################
#######################################$

		else:
			if len(jets) < 2 : continue
			
			#print "len of all particles", len(all_particles)
			all = [a for a in all_particles]

			hg = [a for a in all if a.absPdgId() ==37]
			#print " len of hg", len(hg)
			#print [h.pdgId() for h in hg]
			f_h = []
			for i in range(len(hg)):
				f_h.append(find_child(hg[i]))
			#print " shape of f_h:", np.shape(f_h) 
			# print " 1 higgs child",   f_h[0:1][0][0].pdgId(), f_h[0:1][0][1].pdgId()
			chi_1_plus_child = []
			
			for i in range(len(hg)):
				#print " f_h[1][0] and f_h[1][1] ", f_h[i][0].pdgId() , f_h[i][1].pdgId()
				chi_1_plus_child = find_child(f_h[i][0])
				#print "chi_1_plus_child:", [p.pdgId() for p in chi_1_plus_child]
				
	       		lep_list = [p for p in chi_1_plus_child if p.absPdgId() == 11 or p.absPdgId() ==13]
			
			

			if len(lep_list) ==0: continue
			MT = (lep_list[0].p4()+metvector0).Mt()
			
			#new
			for p in chi_1_plus_child:
				if p.absPdgId() == 11 or p.absPdgId() == 13:
					l1 = p
				if p.absPdgId() == 1000022:
					c0 = p

			print [p.pdgId() for p in f_h[0]]
			#new
			###finding the dR between the ch1 and ch0 from h+
			#for p in range(len())
			# for p in range(len(hg)):
			# 	for c in range(2):
			# 		print "pdgId", f_h[c][p].pdgId()
			# 		print "Pt",f_h[c][p].p4().Pt()
			# 		print "Eta",f_h[c][p].p4().Eta()
			# 		print "Phi",f_h[c][p].p4().Phi()
			# 		print "Mt",f_h[c][p].p4().Mt()
			# 		print "______________"
			#print f_h[0][0].p4().DeltaR(f_h[1][0].p4()) 
			#print c0.p4().DeltaR(l1.p4())
			# dR_c1_c0.Fill(f_h[0][0].p4().DeltaR(f_h[1][0].p4()) )

			# dR_l1_c0.Fill(l1.p4().DeltaR(c0.p4()))

			eta.append(lep_list[0].eta())
			phi.append(lep_list[0].phi())
			eta_jet.append(jets[0].eta())
			phi_jet.append(jets[0].phi())
			phi_met.append(Met.get(0).phi())
			eta_jet_s.append(jets[1].eta())
			phi_jet_s.append(jets[1].phi())

			lepton_pt_l.append(lep_list[0].p4().Pt())
			num_jet_l.append(len(jets))
			pt_leading_jets_l.append(jets[0].p4().Pt())
			pt_second_jets_l.append(jets[1].p4().Pt())
			Met1_l.append(Met.get(1).met())
			Met0_l.append(Met.get(0).met())
			dR_lep_jet.append(lep_list[0].p4().DeltaR(jets[0].p4()))
			MT_l.append(MT)


#LOOK FOR B QUARK OVERLAPING WITH LEADING JETS AND SEE IF THE PARENT OF THE B QUARK 

	# c1 =ROOT.TCanvas()
	# dR_l1_c0.Draw()
	# c1.Print("dR_l1_c0.pdf")
	# dR_c1_c0.Draw()
	# c1.Print("dR_c1_c0.pdf")



	data = list(zip(lepton_pt_l,pt_leading_jets_l,pt_second_jets_l,num_jet_l,Met1_l,Met0_l,MT_l,dR_lep_jet,eta,phi,eta_jet,phi_jet,eta_jet_s,phi_jet_s,phi_met))
	
	
	df = pd.DataFrame(data,columns=["lepton_pt","pt_leading_jets","pt_second_jets","num_jet","Met1","Met0","MT","dR_lep_jet","eta_lepton","phi_lepton","eta_leading_jet","phi_leading_jet","eta_subleading_jet","phi_subleading_jet","phi_met"])
	print df 
	df.to_csv("{}.csv".format("sig1600randseed"))
	#df.to_csv("{}.csv".format(file_chain[k]))
		

	print "Finished"
	break

