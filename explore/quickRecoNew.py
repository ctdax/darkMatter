###  Name: Benjamin Tannenwald
###  Date: July 31, 2020
###  Purpose: class for doing some really basic object selection and event reconstruction. If you want to do other things, you should probably make a new class.

import os, glob
import numpy as np
from matplotlib import pyplot as plt
import uproot, uproot_methods
import uproot_methods.classes.TLorentzVector as TLorentzVector


class quickReco:
    
    def __init__(self, name='', inputTxt='', isTest=False):
        
        self.name = name
        self.inputTxt = inputTxt
        self.isTestRun = isTest
        
        # thresholds
        self.minLepPt  = 15
        self.maxLepEta = 2.5
        self.minJetPt  = 20
        self.maxJetEta = 2.5
    
        # variables unique for each event
        self.selectedMuons = []
        self.selectedElectrons = []
        self.selectedJets = []
        
        # variables to keep track
        self.nJets      = []
        self.nElectrons = []
        self.nMuons     = []
        self.HT         = []
        self.missingET  = []
        self.dileptonMass = []
        self.dileptonPT   = []
        self.dilepton_Jets = []
        self.dilepton_missingET = []
        self.dilepton_HT = []
        self.jetCutflow = {'all':0, 'pt>{}'.format(self.minJetPt):0, '|eta|<{}'.format(self.maxJetEta):0}
        self.electronCutflow = {'all':0, 'pt>{}'.format(self.minLepPt):0, '|eta|<{}'.format(self.maxLepEta):0}
        self.muonCutflow = {'all':0, 'pt>{}'.format(self.minLepPt):0, '|eta|<{}'.format(self.maxLepEta):0}
        self.eventCutflow = {'all':0, 'El: >=2 lep':0, 'El: ==2 OS lep':0, 'El: Z window':0, 'Mu: >=2 lep':0, 'Mu: ==2 OS lep':0, 'Mu: Z window':0}
        
        # Branch Definitions
        self.l_jetPt     = []
        self.l_jetEta    = []
        self.l_jetPhi    = []
        self.l_jetMass   = []
        self.l_elPt      = []
        self.l_elEta     = []
        self.l_elPhi     = []
        self.l_elCharge  = []
        self.l_muPt      = []
        self.l_muEta     = []
        self.l_muPhi     = []
        self.l_muCharge  = []
        self.l_missingET_met  = []
        self.l_missingET_phi  = []
        self.l_scalarHT       = []
        
            
    def process(self):
        """main function for processing files"""
                
        # *** 0. Set TTree of file(s)
        self.setTree()
        
        # *** 1. Loop over events
        for iEvent in range(0, len(self.delphesTree)):
           
            #keep track of events
            if iEvent%500 == 0:
                print("Processed {} events...".format(iEvent))
            
            #skip out if testRun
            if self.isTestRun and iEvent > 10:
                continue
            
            # ** A. Re-initialize collections
            self.selectedElectrons = []
            self.selectedMuons = []
            self.selectedJets = []
            
            # ** B. Object selection
            self.leptonSelection('Electron', iEvent)
            self.leptonSelection('Muon', iEvent)
            self.jetSelection(iEvent)

            # ** C. Event selection
            self.eventSelection(iEvent)
            

       # *** 2. Print some summaries
    
    
    def eventSelection(self, _iEvent):
        """function to select events"""
        
        #print("Evt: {}, N_electrons= {}, N_muons= {}".format(_iEvent, len(self.selectedElectrons), len(self.selectedMuons)))
        self.eventCutflow['all'] += 1
        
        if len(self.selectedElectrons) > 2 and len(self.selectedMuons)>2:
            print(">2 muons AND >2 electrons in Event {}. Ambiguous, so skipping...".format(_iEvent))
            return
        
        # ** 1. electron channel
        self.dileptonReco(self.selectedElectrons, 'electron', _iEvent)

        # ** 2. muon channel
        self.dileptonReco(self.selectedMuons, 'muon', _iEvent)

    def dileptonReco(self, _leptons, flavor, _iEvent):
        """helper function for dilepton reconstruction"""
        
        cutflowTag = 'El' if flavor=='electron' else 'Mu'
        
        if len( _leptons) >= 2:
            self.eventCutflow['{}: >=2 lep'.format(cutflowTag)] += 1
            
            if len(_leptons) == 2:
                lep0 = _leptons[0]
                lep1 = _leptons[1]
        
                if lep0[1]*lep1[1] == -1:
                    self.eventCutflow['{}: ==2 OS lep'.format(cutflowTag)] += 1
                    
                    m_ll = ( lep0[0] + lep1[0] ).mass
                    pt_ll = ( lep0[0] + lep1[0] ).pt
                    
                    self.dileptonMass.append( m_ll)
                    self.dileptonPT.append( pt_ll)
                    self.dilepton_Jets.append(self.nJets[_iEvent])
                    self.dilepton_missingET.append(self.missingET[_iEvent])
                    self.dilepton_HT.append(self.HT[_iEvent])
            
            elif len(_leptons) >2:
                print("Event {} has >2 {}. print (pT, charge)...".format(_iEvent, flavor))
                for lep in _leptons:
                    print("pt: {}, charge: {}".format(lep[0].pt, lep[1]))
                return
            
        else:
            return
            
    def leptonSelection(self, flavor, _iEvent):
        """function for performing lepton selection"""
        
        _selectedLeptons = []
        
        _pt     = self.l_elPt if flavor=='Electron' else self.l_muPt 
        _eta    = self.l_elEta if flavor=='Electron' else self.l_muEta 
        _phi    = self.l_elPhi if flavor=='Electron' else self.l_muPhi
        _charge = self.l_elCharge if flavor=='Electron' else self.l_muCharge 
        _mass   = 0.511 * 1e-3 if flavor == 'Electron' else 105.6*1e-3 # need conversion 1e-3 for MeV to GeV
        
        for iLep in range(0, len(_pt[_iEvent])): 
            if flavor=='Electron':
                self.electronCutflow['all'] += 1
            elif flavor=='Muon':
                self.muonCutflow['all'] += 1
             
            # ** A. Check pt above min threshold
            if _pt[_iEvent][iLep] < self.minLepPt:
                continue
                
            if flavor=='Electron':
                self.electronCutflow['pt>{}'.format(self.minLepPt)] += 1
            elif flavor=='Muon':
                self.muonCutflow['pt>{}'.format(self.minLepPt)] += 1
                
            # ** B. Check eta inside max abs eta range
            if np.abs(_eta[_iEvent][iLep]) > self.maxLepEta:
                continue

            if flavor=='Electron':
                self.electronCutflow['|eta|<{}'.format(self.maxLepEta)] += 1
            elif flavor=='Muon':
                self.muonCutflow['|eta|<{}'.format(self.maxLepEta)] += 1
            
            # ** C. Make LorentzVector and store
            _tlv = TLorentzVector.PtEtaPhiMassLorentzVector( _pt[_iEvent][iLep], _eta[_iEvent][iLep], _phi[_iEvent][iLep], _mass)
            _selectedLeptons.append( (_tlv, _charge[_iEvent][iLep]) )
            
            
        # some global stuff after selection        
        if flavor=='Electron':
            self.selectedElectrons = _selectedLeptons
            self.nElectrons.append( len(_selectedLeptons))
        elif flavor=='Muon':
            self.selectedMuons = _selectedLeptons
            self.nMuons.append( len(_selectedLeptons))
                
                
    def jetSelection(self, _iEvent):
        """function for performing jet selection"""
        
        _selectedJets = []
        
        _pt     = self.l_jetPt  
        _eta    = self.l_jetEta 
        _phi    = self.l_jetPhi
        _mass   = self.l_jetMass

        for iJet in range(0, len(_pt[_iEvent])): 
            self.jetCutflow['all'] += 1
             
            # ** A. Check pt above min threshold
            if _pt[_iEvent][iJet] < self.minJetPt:
                continue
                
            self.jetCutflow['pt>{}'.format(self.minJetPt)] += 1
                
            # ** B. Check eta inside max abs eta range
            if np.abs(_eta[_iEvent][iJet]) > self.maxJetEta:
                continue

            self.jetCutflow['|eta|<{}'.format(self.maxJetEta)] += 1
            
            # ** C. Make LorentzVector and store
            _tlv = TLorentzVector.PtEtaPhiMassLorentzVector( _pt[_iEvent][iJet], _eta[_iEvent][iJet], _phi[_iEvent][iJet], _mass[_iEvent][iJet])
            _selectedJets.append( _tlv )
            
        # some global stuff after selection        
        self.selectedJets = _selectedJets
        self.nJets.append( len(_selectedJets))
       

    def setTree(self):
        """function to determine whether input is a single file or a directory"""
               
        if os.path.isfile( self.inputTxt): # single file
            print("...Loading file \n{}".format(self.inputTxt))
            self.delphesTree = uproot.open(self.inputTxt)['Delphes']
            self.loadBranches()
            
        elif os.path.isdir( self.inputTxt): # directory (hopefully of files)
            rootFiles = [file.split('\n')[0] for file in glob.glob(self.inputTxt+'/*.root')]
            print("...Loading file(s) \n{}".format( '\n'.join(rootFiles) ))
            self.delphesTree = uproot.lazyarrays( path=rootFiles,
                                                  treepath='Delphes',
                                                  branches=['Jet.PT', 'Jet.Eta', 'Jet.Phi', 'Jet.Mass',
                                                           'Electron.PT', 'Electron.Eta', 'Electron.Phi', 'Electron.Charge',
                                                           'Muon.PT', 'Muon.Eta', 'Muon.Phi', 'Muon.Charge',
                                                           'MissingET.MET', 'MissingET.Phi', 'ScalarHT.HT'])
            
            self.loadBranches(isSingleFile=False)
                        
        else: # probably doesn't exist
            print("!!! Input Location {} DNE\nEXITING!\n".format(self.inputTxt))
            exit()
            
        return

    
    def loadBranches(self, isSingleFile=True):
        """function for loading branches from delphes file"""
        
        if isSingleFile:
            # branches
            self.l_jetPt   = uproot.tree.TBranchMethods.array(self.delphesTree['Jet']['Jet.PT']).tolist()
            self.l_jetEta  = uproot.tree.TBranchMethods.array(self.delphesTree['Jet']['Jet.Eta']).tolist()
            self.l_jetPhi  = uproot.tree.TBranchMethods.array(self.delphesTree['Jet']['Jet.Phi']).tolist()
            self.l_jetMass = uproot.tree.TBranchMethods.array(self.delphesTree['Jet']['Jet.Mass']).tolist()
        
            self.l_elPt     = uproot.tree.TBranchMethods.array(self.delphesTree['Electron']['Electron.PT']).tolist()
            self.l_elEta    = uproot.tree.TBranchMethods.array(self.delphesTree['Electron']['Electron.Eta']).tolist()
            self.l_elPhi    = uproot.tree.TBranchMethods.array(self.delphesTree['Electron']['Electron.Phi']).tolist()
            self.l_elCharge = uproot.tree.TBranchMethods.array(self.delphesTree['Electron']['Electron.Charge']).tolist()
       
            self.l_muPt     = uproot.tree.TBranchMethods.array(self.delphesTree['Muon']['Muon.PT']).tolist()
            self.l_muEta    = uproot.tree.TBranchMethods.array(self.delphesTree['Muon']['Muon.Eta']).tolist()
            self.l_muPhi    = uproot.tree.TBranchMethods.array(self.delphesTree['Muon']['Muon.Phi']).tolist()
            self.l_muCharge = uproot.tree.TBranchMethods.array(self.delphesTree['Muon']['Muon.Charge']).tolist()
        
            self.l_missingET_met  = uproot.tree.TBranchMethods.array(self.delphesTree['MissingET']['MissingET.MET']).tolist()
            self.l_missingET_phi  = uproot.tree.TBranchMethods.array(self.delphesTree['MissingET']['MissingET.Phi']).tolist()
            self.l_scalarHT       = uproot.tree.TBranchMethods.array(self.delphesTree['ScalarHT']['ScalarHT.HT']).tolist()
        
        else: # multi-file
            self.l_jetPt   = self.delphesTree['Jet.PT'].tolist()
            self.l_jetEta  = self.delphesTree['Jet.Eta'].tolist()
            self.l_jetPhi  = self.delphesTree['Jet.Phi'].tolist()
            self.l_jetMass = self.delphesTree['Jet.Mass'].tolist()
        
            self.l_elPt     = self.delphesTree['Electron.PT'].tolist()
            self.l_elEta    = self.delphesTree['Electron.Eta'].tolist()
            self.l_elPhi    = self.delphesTree['Electron.Phi'].tolist()
            self.l_elCharge = self.delphesTree['Electron.Charge'].tolist()
       
            self.l_muPt     = self.delphesTree['Muon.PT'].tolist()
            self.l_muEta    = self.delphesTree['Muon.Eta'].tolist()
            self.l_muPhi    = self.delphesTree['Muon.Phi'].tolist()
            self.l_muCharge = self.delphesTree['Muon.Charge'].tolist()
        
            self.l_missingET_met  = self.delphesTree['MissingET.MET'].tolist()
            self.l_missingET_phi  = self.delphesTree['MissingET.Phi'].tolist()
            self.l_scalarHT       = self.delphesTree['ScalarHT.HT'].tolist()
            
        print("...Finished loading branches")
