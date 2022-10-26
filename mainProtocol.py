"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Program Description:
    
    Perform signal processing to filter blood pulse peaks. Seperate the peaks,
    and extract key features from each pulse. 
    
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        pip install -U numpy scikit-learn matplotlib tensorflow openpyxl pyserial joblib pandas
        pip install -U natsort pyexcel shap seaborn findpeaks keras
        pip install -U pyexcel pyexcel-xls pyexcel-xlsx BaselineRemoval peakutils lmfit scikit-image
        
    --------------------------------------------------------------------------
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import shutil
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from natsort import natsorted
# Import Data Extraction Files (And Their Location)
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with All the Helper Files
import excelProcessing

# Import Analysis Files (And Their Locations)
sys.path.append('./Data Aquisition and Analysis/_Analysis Protocols')  # Folder with All the Helper Files
sys.path.append('./Helper Files/Data Aquisition and Analysis/_Analysis Protocols')  # Folder with All the Helper Files
import gsrAnalysis
import pulseAnalysis
import chemicalAnalysis
import temperatureAnalysis

# Import Machine Learning Files (And They Location)
sys.path.append("./Helper Files/Machine Learning/")
import machineLearningMain   # Class Header for All Machine Learning
import featureAnalysis       # Functions for Feature Analysis


from numpy import arange

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

N = 17
data = np.arange(N +1)


# =================
## custom colormap:

# red-green colormap:
cdict = {'blue':   [(0.0,  0.0, 0.0),
                   (0.5,  1.0, 0),
                   (1.0,  1.0, 1.0)],

         'red': [(0.0,  0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.75, 1.0, 0.5),
                   (1.0,  0.5, 0)],

         'green':  [(0.0,  0.0, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  1.0, 0)]}

red_green_cm = LinearSegmentedColormap('RedGreen', cdict, N)

colors = cm.get_cmap(red_green_cm, N)

# fig = plt.figure()

# basinhopping
# ampgo
# cg


# bfgs


# -------------------------------------------------------------------------- #
# --------------------------- Program Starts Here -------------------------- #

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #    

    # Analysis Parameters
    timePermits = False                      # Construct Plots that Take a Long TIme
    plotFeatures = False                     # Plot the Analyzed Features
    trackRawData = False
    stimulusTimes = [1000, 1000 + 60*3]      # The [Beginning, End] of the Stimulus in Seconds; Type List.
    stimulusTimes_Delayed = [1500, 1500 + 60*3]     # The [Beginning, End] of the Stimulus in Seconds; Type List.
    
    # Specify Which Signals to Use
    extractGSR = True
    extractPulse = True
    extractChemical = True
    extractTemperature = True
    # Reanalyze Peaks from Scratch (Don't Use Saved Features)
    reanalyzeData_GSR = False
    reanalyzeData_Pulse = False
    reanalyzeData_Chemical = False    
    reanalyzeData_Temperature = False

    # Specify the Unit of Data for Each 
    unitOfData_GSR = "micro"                # Specify the Unit the Data is Represented as: ['', 'milli', 'micro', 'nano', 'pico', 'fempto']
    unitOfData_Pulse = "pico"               # Specify the Unit the Data is Represented as: ['', 'milli', 'micro', 'nano', 'pico', 'fempto']
    unitOfData_Temperature = ""             # Specify the Unit the Data is Represented as: ['', 'milli', 'micro', 'nano', 'pico', 'fempto']
    unitOfData_Chemical_ISE = "milli"       # Specify the Unit the Data is Represented as: ['', 'milli', 'micro', 'nano', 'pico', 'fempto']
    unitOfData_Chemical_Enzym = "micro"     # Specify the Unit the Data is Represented as: ['', 'milli', 'micro', 'nano', 'pico', 'fempto']

    # Specify the Location of the Subject Files
    dataFolderWithSubjects = './Input Data/Current Analysis/'  # Path to ALL the Subject Data. The Path Must End with '/'
    compiledFeatureNamesFolder = "./Helper Files/Machine Learning/Compiled Feature Names/All Features/"

    # Specify the Stressors/Sensors Used in this Experiment
    listOfStressors = ['cpt', 'exercise', 'vr']                # This Keyword MUST be Present in the Filename
    listOfSensors = ['pulse', 'ise', 'enzym', 'gsr', 'temp']   # This Keyword MUST be Present in the Filename
    
    # Remove any of the scores: cpt, exercise, vr
    removeScores = [[], [], []]
    
    # ---------------------------------------------------------------------- #
    # ------------------------- Preparation Steps -------------------------- #
    
    # Create Instance of Excel Processing Methods
    excelProcessingGSR = excelProcessing.processGSRData()
    excelProcessingPulse = excelProcessing.processPulseData()
    excelProcessingChemical = excelProcessing.processChemicalData()
    excelProcessingTemperature = excelProcessing.processTemperatureData()

    # Create Instances of all Analysis Protocols
    gsrAnalysisProtocol = gsrAnalysis.signalProcessing(stimulusTimes)
    pulseAnalysisProtocol = pulseAnalysis.signalProcessing()
    chemicalAnalysisProtocol = chemicalAnalysis.signalProcessing(plotData = True)
    temperatureAnalysisProtocol = temperatureAnalysis.signalProcessing(stimulusTimes)

    subjectFolderPaths = []
    # Extract the Files for from Each Subject
    for folder in os.listdir(dataFolderWithSubjects):
        if 'subject' not in folder.lower() and not folder.startswith(("$", '~', '_')):
            continue
        
        subjectFolder = dataFolderWithSubjects + folder + "/"
        # Check Whether the Path is a Folder (Exclude Cache Folders)
        if os.path.isdir(subjectFolder):
                # Save the Folder's path for Later Analysis
                subjectFolderPaths.append(subjectFolder)
    # Sort the Folders for Ease in Debugging
    subjectFolderPaths = natsorted(subjectFolderPaths)

    # Define Map of Units to Scale Factors
    scaleFactorMap = {'': 1, 'milli': 10**-3, 'micro': 10**-6, 'nano': 10**-9, 'pico': 10**-12, 'fempto': 10**-15}
    # Find the Scale Factor for the Data
    scaleFactor_GSR = 1 #scaleFactorMap[unitOfData_GSR]
    scaleFactor_Pulse = scaleFactorMap[unitOfData_Pulse] # Pulse 10**-12 
    scaleFactor_Temperature = 1 #scaleFactorMap[unitOfData_Temperature]
    scaleFactor_Chemical_ISE = 1 #scaleFactorMap[unitOfData_Chemical_ISE]
    scaleFactor_Chemical_Enzym = 1 #scaleFactorMap[unitOfData_Chemical_Enzym]
    
    # ---------------------------------------------------------------------- #
    # ------------------------ Specify the Features ------------------------ #
        
    # Compile Features
    scoreFeatureLabels = []  # Compile Stress Scores
    
    if extractPulse:
        # Specify the Paths to the Pulse Feature Names
        pulseFeaturesFile_StressLevel = compiledFeatureNamesFolder + "pulseFeatureNames_StressLevel.txt"
        pulseFeaturesFile_SignalIncrease = compiledFeatureNamesFolder + "pulseFeatureNames_SignalIncrease.txt"
        # Extract the Pulse Feature Names we are Using
        pulseFeatureNames_StressLevel = excelProcessingPulse.extractFeatureNames(pulseFeaturesFile_StressLevel, prependedString = "pulseFeatures.extend([", appendToName = "_StressLevel")[1:]
        pulseFeatureNames_SignalIncrease = excelProcessingPulse.extractFeatureNames(pulseFeaturesFile_SignalIncrease, prependedString = "pulseFeatures.extend([", appendToName = "_SignalIncrease")[1:]
        # Combine all the Features
        pulseFeatureNames = []
        pulseFeatureNames.extend(pulseFeatureNames_StressLevel)
        pulseFeatureNames.extend(pulseFeatureNames_SignalIncrease)
        # Get Pulse Names Without Anything Appended
        pulseFeatureNamesFull = excelProcessingPulse.extractFeatureNames(pulseFeaturesFile_SignalIncrease, prependedString = "pulseFeatures.extend([", appendToName = "")
        pulseFeatureNamesFull.extend(excelProcessingPulse.extractFeatureNames(pulseFeaturesFile_StressLevel, prependedString = "pulseFeatures.extend([", appendToName = "")[1:])
        # Create Data Structure to Hold the Features
        pulseFeatures = []
        pulseFeatureLabels = []  
        # Track one pulse feature
        rawPulseFeatureData = []
    
    if extractChemical:
        # Specify the Paths to the Chemical Feature Names
        glucoseFeaturesFile = compiledFeatureNamesFolder + "glucoseFeatureNames.txt"
        lactateFeaturesFile = compiledFeatureNamesFolder + "lactateFeatureNames.txt"
        uricAcidFeaturesFile = compiledFeatureNamesFolder + "uricAcidFeatureNames.txt"
        # Extract the Chemical Feature Names we are Using
        glucoseFeatureNames = excelProcessingChemical.extractFeatureNames(glucoseFeaturesFile, prependedString = "peakFeatures.extend([", appendToName = '_Glucose')
        lactateFeatureNames = excelProcessingChemical.extractFeatureNames(lactateFeaturesFile, prependedString = "peakFeatures.extend([", appendToName = '_Lactate', )
        uricAcidFeatureNames = excelProcessingChemical.extractFeatureNames(uricAcidFeaturesFile, prependedString = "peakFeatures.extend([", appendToName = '_UricAcid', )
        # Combine all the Features
        chemicalFeatureNames_Enzym = []
        chemicalFeatureNames_Enzym.extend(glucoseFeatureNames)
        chemicalFeatureNames_Enzym.extend(lactateFeatureNames)
        chemicalFeatureNames_Enzym.extend(uricAcidFeatureNames)
        # Create Data Structure to Hold the Features
        chemicalFeatures_Enzym = []
        chemicalFeatureLabels_Enzym = []
        # Track raw data
        rawEnzymData = []
        
        # Specify the Paths to the Chemical Feature Names
        sodiumFeaturesFile = compiledFeatureNamesFolder + "sodiumFeatureNames.txt"
        potassiumFeaturesFile = compiledFeatureNamesFolder + "potassiumFeatureNames.txt"
        ammoniumFeaturesFile = compiledFeatureNamesFolder + "ammoniumFeatureNames.txt"
        # Extract the Chemical Feature Names we are Using
        sodiumFeatureNames = excelProcessingChemical.extractFeatureNames(sodiumFeaturesFile, prependedString = "peakFeatures.extend([", appendToName = '_Sodium')
        potassiumFeatureNames = excelProcessingChemical.extractFeatureNames(potassiumFeaturesFile, prependedString = "peakFeatures.extend([", appendToName = '_Potassium', )
        ammoniumFeatureNames = excelProcessingChemical.extractFeatureNames(ammoniumFeaturesFile, prependedString = "peakFeatures.extend([", appendToName = '_Ammonium', )
        # Combine all the Features
        chemicalFeatureNames_ISE = []
        chemicalFeatureNames_ISE.extend(sodiumFeatureNames)
        chemicalFeatureNames_ISE.extend(potassiumFeatureNames)
        chemicalFeatureNames_ISE.extend(ammoniumFeatureNames)
        # Create Data Structure to Hold the Features
        chemicalFeatures_ISE = []
        chemicalFeatureLabels_ISE = []
        # Track raw data
        rawISEData = []
        
    if extractGSR:
        # Specify the Paths to the GSR Feature Names
        gsrFeaturesFile = compiledFeatureNamesFolder + "gsrFeatureNames.txt"
        # Extract the GSR Feature Names we are Using
        gsrFeatureNames = excelProcessingGSR.extractFeatureNames(gsrFeaturesFile, prependedString = "gsrFeatures.extend([", appendToName = '_GSR')
        # Create Data Structure to Hold the Features
        gsrFeatures = []
        gsrFeatureLabels = []
        # Track raw data
        rawGSRData = []
        
    if extractTemperature:
        # Specify the Paths to the Temperature Feature Names
        temperatureFeaturesFile = compiledFeatureNamesFolder + "temperatureFeatureNames.txt"
        # Extract the GSR Feature Names we are Using
        temperatureFeatureNames = excelProcessingTemperature.extractFeatureNames(temperatureFeaturesFile, prependedString = "temperatureFeatures.extend([", appendToName = '_Temperature')
        # Create Data Structure to Hold the Features
        temperatureFeatures = []
        temperatureFeatureLabels = []
        # Track raw data
        rawTempData = []
        
    # ---------------------------------------------------------------------- #
    # -------------------- Data Collection and Analysis -------------------- #
    
    # Loop Through Each Subject
    for subjectFolder in subjectFolderPaths:
        
        # CPT Score
        cptScore = subjectFolder.split("CPT")
        if len(cptScore) == 1:
            cptScore = None
        else:
            cptScore = int(cptScore[1][0:2])
        # Excersize Score
        exerciseScore = subjectFolder.split("Exercise")
        if len(exerciseScore) == 1:
            exerciseScore = subjectFolder.split("Exer")
        if len(exerciseScore) == 1:
            exerciseScore = None
        else:
            exerciseScore = int(exerciseScore[1][0:2])
        # CPT Score
        vrScore = subjectFolder.split("VR")
        if len(vrScore) == 1:
            vrScore = None
        else:
            vrScore = int(vrScore[1][0:2])
            
        # Remove bad scores
        if cptScore in removeScores[0]:
            cptScore = None
        if exerciseScore in removeScores[1]:
            exerciseScore = None
        if vrScore in removeScores[2]:
            vrScore = None
            
        # Label the Score of the File
        scoreLabels_OneTrial = [cptScore, exerciseScore, vrScore]
        scoreFeatureLabels.extend([None]*len(scoreLabels_OneTrial))
                        
        # ------- Organize the Files within Each Stressor and Sensor ------- #
        
        # Organize/Label the Files in the Folder: Pulse, Chemical, GSR, Temp
        fileMap = [[None for _ in  range(len(listOfSensors))] for _ in range(len(listOfStressors))]
        # Loop Through Each File and Label the Stressor Analyzed
        for file in os.listdir(subjectFolder):
            if file.startswith(("#","~","$")):
                continue

            # Find the Type of Stressor in the File
            for stressorInd, stressor in enumerate(listOfStressors):
                if stressor.lower() in file.lower():
                    
                    # Extract the Stress Information from the Filename
                    if scoreFeatureLabels[stressorInd - len(listOfStressors)] == None:
                        scoreFeatureLabels[stressorInd - len(listOfStressors)] = scoreLabels_OneTrial[stressorInd]
                
                    # Find the Type of Sensor the File Used
                    for sensorInd, sensor in enumerate(listOfSensors):
                        if sensor.lower() in file[len(stressor):].lower():
                            
                            # Organize the Files by Their Stressor and Sensor Type 
                            if sensor == "pulse":
                                fileMap[stressorInd][sensorInd] = subjectFolder + file + "/";
                            else:
                                fileMap[stressorInd][sensorInd] = subjectFolder + file;
                            break
                    break
        fileMap = np.array(fileMap)
        
        # Quick check if files are found (Alert user if not)
        for stressorFiles in fileMap:
            if None in stressorFiles:
                print("\n\nNot all files found here:\n", stressorFiles, "\n\n")
                
        # ------------------------- Pulse Analysis ------------------------- #
        
        if extractPulse:
            # Extract the Pulse Folder Paths in the Map
            pulseFolders = fileMap[:, listOfSensors.index("pulse")]
            
            # Loop Through the Pulse Data for Each Stressor
            for featureLabel, pulseFolder in enumerate(pulseFolders):
                if pulseFolder == None:
                    # Compile the Featues into One Array
                    pulseFeatureLabels.append(None)
                    pulseFeatures.append([None]*len(pulseFeatureNames))
                    continue
                
                savePulseDataFolder = pulseFolder + "Pulse Analysis/"    # Data Folder to Save the Data; MUST END IN '/'
                if not reanalyzeData_Pulse and os.path.isfile(pulseFolder + "/Pulse Analysis/Compiled Data in Excel/Feature List.xlsx"):
                    pulseFeatureList_Full = excelProcessingPulse.getSavedFeatures(pulseFolder + "/Pulse Analysis/Compiled Data in Excel/Feature List.xlsx")
                    featureTimes = pulseFeatureList_Full[:,0]
                    pulseFeatureListExact = pulseFeatureList_Full[:,1:]
                    
                    numSecondsAverage = 30; pulseFeatureList = []
                    # Calculate the Running Average
                    for timePointInd in range(len(featureTimes)):
                        currentTimePoint = featureTimes[timePointInd]
                        
                        # Get the Interval of Feature and Take the Average
                        featureInterval = pulseFeatureListExact[0:timePointInd+1][featureTimes[0:timePointInd+1] > currentTimePoint - numSecondsAverage]
                        pulseFeatureList.append(stats.trim_mean(featureInterval, 0.3))
                    pulseFeatureList = np.array(pulseFeatureList)

                else:
                    pulseExcelFiles = []
                    # Collect all the Pulse Files for the Stressor
                    for file in os.listdir(pulseFolder):
                        file = file.decode("utf-8") if type(file) == type(b'') else file
                        if file.endswith(("xlsx", "xls")) and not file.startswith(("$", '~')):
                            pulseExcelFiles.append(pulseFolder + file)
                    pulseExcelFiles = natsorted(pulseExcelFiles)
                
                    # Loop Through Each Pulse File
                    pulseAnalysisProtocol.resetGlobalVariables()
                    for pulseExcelFile in pulseExcelFiles:
                        
                        # Read Data from Excel
                        time, signalData = excelProcessingPulse.getData(pulseExcelFile, testSheetNum = 0)
                        signalData = signalData*scaleFactor_Pulse
                                        
                        # Calibrate Systolic and Diastolic Pressure
                        fileBasename = os.path.basename(pulseExcelFile)
                        pressureInfo = fileBasename.split("SYS")
                        if len(pressureInfo) > 1 and pulseAnalysisProtocol.systolicPressure0 == None:
                            pressureInfo = pressureInfo[-1].split(".")[0]
                            systolicPressure0, diastolicPressure0 = pressureInfo.split("_DIA")
                            pulseAnalysisProtocol.setPressureCalibration(float(systolicPressure0), float(diastolicPressure0))
                        
                        # Check Whether the StartTime is Specified in the File
                        if fileBasename.lower() in ["cpt", "exer", "vr", "start"] and not stimulusTimes[0]:
                            stimulusTimes[0] = pulseAnalysisProtocol.timeOffset
                                                    
                        # Seperate Pulses, Perform Indivisual Analysis, and Extract Features
                        pulseAnalysisProtocol.analyzePulse(time, signalData, minBPM = 30, maxBPM = 180)
                    
                    # Remove Previous Analysis if Present
                    if os.path.isdir(savePulseDataFolder):
                        shutil.rmtree(savePulseDataFolder)
                    pulseAnalysisProtocol.featureListExact = np.array(pulseAnalysisProtocol.featureListExact)
                    # Save the Features and Filtered Data
                    saveCompiledDataPulse = savePulseDataFolder + "Compiled Data in Excel/"
                    excelProcessingPulse.saveResults(pulseAnalysisProtocol.featureListExact, pulseFeatureNamesFull, saveCompiledDataPulse, "Feature List.xlsx", sheetName = "Pulse Features")
                    excelProcessingPulse.saveFilteredData(pulseAnalysisProtocol.time, pulseAnalysisProtocol.signalData, pulseAnalysisProtocol.filteredData, saveCompiledDataPulse, "Filtered Data.xlsx", "Filtered Data")
                    
                    # Compile the Features from the Data
                    featureTimes = pulseAnalysisProtocol.featureListExact[:,0]
                    pulseFeatureList = np.array(pulseAnalysisProtocol.featureListAverage)
                    pulseFeatureListExact = pulseAnalysisProtocol.featureListExact[:,1:]
                    # Assert That There are Equal Features and Feature Times
                    assert len(featureTimes) == len(pulseFeatureList)
    
                # Quick Check that All Points Have the Correct Number of Features
                for feature in pulseFeatureList:
                    assert len(feature) == len(pulseFeatureNames)
                
                # Plot the Features in Time
                if timePermits:
                    plotPulseFeatures = featureAnalysis.featureAnalysis(featureTimes, pulseFeatureListExact, pulseFeatureNamesFull[1:], stimulusTimes, savePulseDataFolder)
                    plotPulseFeatures.singleFeatureAnalysis()   
                    
                # Downsize the Features into One Data Point
                # ********************************
                # FInd the Indices of the Stimuli
                startStimulusInd = np.argmin(abs(featureTimes - stimulusTimes[0]))
                endStimulusInd = np.argmin(abs(featureTimes - stimulusTimes[1]))
                
                # Caluclate the Baseline/Stress Levels
                restValues = stats.trim_mean(pulseFeatureList[ int(startStimulusInd/6):int(2*startStimulusInd/4),:], 0.4)
                stressValues = stats.trim_mean(pulseFeatureList[ int((endStimulusInd+startStimulusInd)/2) :endStimulusInd,: ], 0.4)
                stressElevation = stressValues - restValues
                # Calculate the Stress Rise/Fall
                stressSlopes = np.polyfit(featureTimes[startStimulusInd:endStimulusInd], pulseFeatureList[ startStimulusInd:endStimulusInd,: ], 1)[0]
    
                # Organize the Signals
                pulseFeatures_StressLevel = stressValues[0:len(pulseFeatureNames_StressLevel)]
                pulseFeatures_SignalIncrease = stressElevation[len(pulseFeatureNames_StressLevel):]
                # Compile the Signals
                subjectPulseFeatures = []
                subjectPulseFeatures.extend(pulseFeatures_StressLevel)
                subjectPulseFeatures.extend(pulseFeatures_SignalIncrease)
                # Assert the Number of Signals are Correct
                assert len(subjectPulseFeatures) == len(pulseFeatureNames)
                assert len(pulseFeatures_StressLevel) == len(pulseFeatureNames_StressLevel)
                assert len(pulseFeatures_SignalIncrease) == len(pulseFeatureNames_SignalIncrease)
                # ********************************
                
                # Compile the Featues into One Array
                pulseFeatureLabels.append(featureLabel)
                pulseFeatures.append(subjectPulseFeatures)
                
                if trackRawData:
                    # Track raw feature
                    rawPulseFeatureData.append([
                            featureTimes, 
                            # pulseFeatureList[:, pulseFeatureNames.index("centralAugmentationIndex_EST_SignalIncrease")]
                            # pulseFeatureList[:, pulseFeatureNames.index("reflectionIndex_SignalIncrease")]
                            # pulseFeatureList[:, pulseFeatureNames.index("systolicDicroticNotchAmpRatio_SignalIncrease")]
                            # pulseFeatureList[:, pulseFeatureNames.index("dicroticRiseVelMaxTime_StressLevel")]
                            pulseFeatureList[:, pulseFeatureNames.index("dicroticNotchDicroticAccelRatio_SignalIncrease")]
                            
                        ])
    
        # ------------------------ Chemical Analysis ----------------------- #
        
        if extractChemical:
            # Extract the Chemical Folder Paths in the Map
            chemicalFiles_ISE = fileMap[:, listOfSensors.index("ise")]
            chemicalFiles_Enzym = fileMap[:, listOfSensors.index("enzym")]
            
            # --------------- Enzym
            # Loop Through the Pulse Data for Each Stressor
            for featureLabel, chemicalFile in enumerate(chemicalFiles_Enzym):
                if chemicalFile == None:
                    chemicalFeatureLabels_Enzym.append(None)
                    chemicalFeatures_Enzym.append([None]*len(chemicalFeatureNames_Enzym))
                    continue
                # Extract the Specific Chemical Filename
                chemicalFilename = os.path.basename(chemicalFile[:-1]).split(".")[0]
                saveCompiledDataChemical = subjectFolder + "Chemical Analysis/Compiled Data in Excel/" + chemicalFilename + "/"
    
                if not reanalyzeData_Chemical and os.path.isfile(saveCompiledDataChemical + "Feature List.xlsx"):
                    subjectChemicalFeatures = excelProcessingChemical.getSavedFeatures(saveCompiledDataChemical + "Feature List.xlsx")[0]
                    
                    # Organize the Features of Enzymatic
                    glucoseFeatures = subjectChemicalFeatures[0:len(glucoseFeatureNames)]
                    lactateFeatures = subjectChemicalFeatures[len(glucoseFeatureNames):len(glucoseFeatureNames) + len(lactateFeatureNames)]
                    uricAcidFeatures = subjectChemicalFeatures[len(lactateFeatureNames) + len(glucoseFeatureNames):]
                    
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectChemicalFeatures) == len(chemicalFeatureNames_Enzym)
                    assert len(glucoseFeatures) == len(glucoseFeatureNames)
                    assert len(lactateFeatures) == len(lactateFeatureNames)
                    assert len(uricAcidFeatures) == len(uricAcidFeatureNames)
                else:
                    # Read in the Chemical Data from Excel
                    timePoints, chemicalData = excelProcessingChemical.getData(chemicalFile, testSheetNum = 0)
                    glucose, lactate, uricAcid = chemicalData*scaleFactor_Chemical_Enzym # Extract the Specific Chemicals
                    lactate = lactate*1000 # Correction on Lactate Data
                    
                    # Cull Subjects with Missing Data
                    if len(glucose) == 0 or len(lactate) == 0 or len(uricAcid) == 0:
                        print("Missing Chemical Data in Folder:", subjectFolder)
                        sys.exit()
            
                    # Compile the Features from the Data
                    chemicalAnalysisProtocol.resetGlobalVariables(stimulusTimes_Delayed, saveCompiledDataChemical)
                    chemicalAnalysisProtocol.analyzeChemicals(timePoints, [glucose, lactate, uricAcid], ['glucose', 'lactate', 'uricAcid'], featureLabel)
                    # Get the ChemicalFeatures
                    glucoseFeatures = chemicalAnalysisProtocol.chemicalFeatures['glucoseFeatures'][0][0]
                    lactateFeatures = chemicalAnalysisProtocol.chemicalFeatures['lactateFeatures'][0][0]
                    uricAcidFeatures = chemicalAnalysisProtocol.chemicalFeatures['uricAcidFeatures'][0][0]
                    chemicalAnalysisProtocol.resetGlobalVariables(stimulusTimes_Delayed)
                    # Verify that Features were Found in for All Chemicals
                    if len(glucoseFeatures) == 0 or len(lactateFeatures) == 0 or len(uricAcidFeatures) == 0:
                        print("No Features Found in Some Chemical Data in Folder:", subjectFolder)
                        sys.exit()   
                        
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(glucoseFeatures) == len(glucoseFeatureNames)
                    assert len(lactateFeatures) == len(lactateFeatureNames)
                    assert len(uricAcidFeatures) == len(uricAcidFeatureNames)
                    
                    # Organize the Chemical Features
                    subjectChemicalFeatures = []
                    subjectChemicalFeatures.extend(glucoseFeatures)
                    subjectChemicalFeatures.extend(lactateFeatures)
                    subjectChemicalFeatures.extend(uricAcidFeatures)
                    
                    # Save the Features and Filtered Data
                    excelProcessingChemical.saveResults([subjectChemicalFeatures], chemicalFeatureNames_Enzym, saveCompiledDataChemical, "Feature List.xlsx", sheetName = "Chemical Features")
                
                # Compile the Featues into One Array
                chemicalFeatureLabels_Enzym.append(featureLabel)
                chemicalFeatures_Enzym.append(subjectChemicalFeatures)
                
                if trackRawData:
                    # Read in the Chemical Data from Excel
                    timePoints, chemicalData = excelProcessingChemical.getData(chemicalFile, testSheetNum = 0)
                    glucose, lactate, uricAcid = chemicalData*scaleFactor_Chemical_Enzym # Extract the Specific Chemicals
                    lactate = lactate*1000 # Correction on Lactate Data
                    # Track raw feature
                    rawEnzymData.append([timePoints, glucose, lactate, uricAcid])
            
            # --------------- ISE
            # Loop Through the Pulse Data for Each Stressor
            for featureLabel, chemicalFile in enumerate(chemicalFiles_ISE):
                if chemicalFile == None:
                    chemicalFeatureLabels_ISE.append(None)
                    chemicalFeatures_ISE.append([None]*len(chemicalFeatureNames_ISE))
                    continue
                # Extract the Specific Chemical Filename
                chemicalFilename = os.path.basename(chemicalFile[:-1]).split(".")[0]
                saveCompiledDataChemical = subjectFolder + "Chemical Analysis/Compiled Data in Excel/" + chemicalFilename + "/"
    
                if not reanalyzeData_Chemical and os.path.isfile(saveCompiledDataChemical + "Feature List.xlsx"):
                    subjectChemicalFeatures = excelProcessingChemical.getSavedFeatures(saveCompiledDataChemical + "Feature List.xlsx")[0]
                    
                    # Organize the Features of ISE
                    sodiumFeatures = subjectChemicalFeatures[0:len(sodiumFeatureNames)]
                    potassiumFeatures = subjectChemicalFeatures[len(sodiumFeatureNames):len(sodiumFeatureNames) + len(potassiumFeatureNames)]
                    ammoniumFeatures = subjectChemicalFeatures[len(sodiumFeatureNames) + len(potassiumFeatureNames):]
                    
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectChemicalFeatures) == len(chemicalFeatureNames_ISE)
                    assert len(sodiumFeatures) == len(sodiumFeatureNames)
                    assert len(potassiumFeatures) == len(potassiumFeatureNames)
                    assert len(ammoniumFeatures) == len(ammoniumFeatureNames)
                else:
                    # Read in the Chemical Data from Excel
                    timePoints, chemicalData = excelProcessingChemical.getData(chemicalFile, testSheetNum = 0)
                    sodium, potassium, ammonium = chemicalData*scaleFactor_Chemical_ISE # Extract the Specific Chemicals
                                        
                    # Cull Subjects with Missing Data
                    if len(sodium) == 0 or len(potassium) == 0 or len(ammonium) == 0:
                        print("Missing Chemical Data in Folder:", subjectFolder)
                        sys.exit()
            
                    # Compile the Features from the Data
                    chemicalAnalysisProtocol.resetGlobalVariables(stimulusTimes_Delayed, saveCompiledDataChemical)
                    chemicalAnalysisProtocol.analyzeChemicals(timePoints, [sodium, potassium, ammonium], ['sodium', 'potassium', 'ammonium'], featureLabel, iseData = True)
                    # Get the ChemicalFeatures
                    sodiumFeatures = chemicalAnalysisProtocol.chemicalFeatures['sodiumFeatures'][0]
                    potassiumFeatures = chemicalAnalysisProtocol.chemicalFeatures['potassiumFeatures'][0]
                    ammoniumFeatures = chemicalAnalysisProtocol.chemicalFeatures['ammoniumFeatures'][0]
                    # Verify that Features were Found in for All Chemicals
                    if len(sodiumFeatures) == 0 or len(potassiumFeatures) == 0 or len(ammoniumFeatures) == 0:
                        print("No Features Found in Some Chemical Data in Folder:", subjectFolder)
                        sys.exit()   
                    chemicalAnalysisProtocol.resetGlobalVariables(stimulusTimes_Delayed)
                        
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(sodiumFeatures) == len(sodiumFeatureNames)
                    assert len(potassiumFeatures) == len(potassiumFeatureNames)
                    assert len(ammoniumFeatures) == len(ammoniumFeatureNames)
                    
                    # Organize the Chemical Features
                    subjectChemicalFeatures = []
                    subjectChemicalFeatures.extend(sodiumFeatures)
                    subjectChemicalFeatures.extend(potassiumFeatures)
                    subjectChemicalFeatures.extend(ammoniumFeatures)
                    
                    # Save the Features and Filtered Data
                    excelProcessingChemical.saveResults([subjectChemicalFeatures], chemicalFeatureNames_ISE, saveCompiledDataChemical, "Feature List.xlsx", sheetName = "Chemical Features")
                
                # Compile the Featues into One Array
                chemicalFeatureLabels_ISE.append(featureLabel)
                chemicalFeatures_ISE.append(subjectChemicalFeatures)
                
                if trackRawData:
                    # Read in the Chemical Data from Excel
                    timePoints, chemicalData = excelProcessingChemical.getData(chemicalFile, testSheetNum = 0)
                    sodium, potassium, ammonium = chemicalData*scaleFactor_Chemical_ISE # Extract the Specific Chemicals
                    # Track raw feature
                    rawISEData.append([timePoints, sodium, potassium, ammonium])
                        
        # -------------------------- GSR Analysis -------------------------- #
        
        if extractGSR:
            colorList = ['k', 'tab:blue', 'tab:red']
            # Extract the Pulse Folder Paths in the Map
            gsrFiles = fileMap[:, listOfSensors.index("gsr")]
            # Loop Through the Pulse Data for Each Stressor
            for featureLabel, gsrFile in enumerate(gsrFiles):
                if gsrFile == None:
                    gsrFeatureLabels.append(None)
                    gsrFeatures.append([None]*len(gsrFeatureNames))
                    continue
                # Extract the Specific GSR Filename
                gsrFilename = os.path.basename(gsrFile[:-1]).split(".")[0]
                saveCompiledDataGSR = subjectFolder + "GSR Analysis/Compiled Data in Excel/" + gsrFilename + "/"
    
                if not reanalyzeData_GSR and os.path.isfile(saveCompiledDataGSR + "Feature List.xlsx"):
                    subjectGSRFeatures = excelProcessingChemical.getSavedFeatures(saveCompiledDataGSR + "Feature List.xlsx")[0]
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectGSRFeatures) == len(gsrFeatureNames)
                else:
                    # Read in the GSR Data from Excel
                    excelDataGSR = excelProcessing.processGSRData()
                    timePoints, currentGSR = excelDataGSR.getData(gsrFile, testSheetNum = 0, method = "processed")
                    currentGSR = currentGSR*scaleFactor_GSR # Get Data into micro-Ampes

                    # Process the Data
                    subjectGSRFeatures = gsrAnalysisProtocol.analyzeGSR(timePoints, currentGSR)
                    
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectGSRFeatures) == len(gsrFeatureNames)
                    
                    # Save the Features and Filtered Data
                    excelProcessingGSR.saveResults([subjectGSRFeatures], gsrFeatureNames, saveCompiledDataGSR, "Feature List.xlsx", sheetName = "GSR Features")

                # Compile the Featues into One Array
                gsrFeatureLabels.append(featureLabel)
                gsrFeatures.append(subjectGSRFeatures)
                
                if trackRawData:
                    # Read in the GSR Data from Excel
                    excelDataGSR = excelProcessing.processGSRData()
                    timePoints, currentGSR = excelDataGSR.getData(gsrFile, testSheetNum = 0, method = "processed")
                    currentGSR = currentGSR*scaleFactor_GSR # Get Data into micro-Ampes
                    # Track raw feature
                    rawGSRData.append([timePoints, currentGSR])  
                    
        # ---------------------- Temperature Analysis ---------------------- #
        
        if extractTemperature:
            # Extract the Pulse Folder Paths in the Map
            temperatureFiles = fileMap[:, listOfSensors.index("temp")]
            # Loop Through the Pulse Data for Each Stressor
            for featureLabel, temperatureFile in enumerate(temperatureFiles):
                if temperatureFile == None:
                    temperatureFeatureLabels.append(None)
                    temperatureFeatures.append([None]*len(temperatureFeatureNames))
                    continue
                # Extract the Specific temperature Filename
                temperatureFilename = os.path.basename(temperatureFile[:-1]).split(".")[0]
                saveCompiledDataTemperature = subjectFolder + "temperature Analysis/Compiled Data in Excel/" + temperatureFilename + "/"
    
                if not reanalyzeData_Temperature and os.path.isfile(saveCompiledDataTemperature + "Feature List.xlsx"):
                    subjectTemperatureFeatures = excelProcessingChemical.getSavedFeatures(saveCompiledDataTemperature + "Feature List.xlsx")[0]
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectTemperatureFeatures) == len(temperatureFeatureNames)
                else:
                    # Read in the temperature Data from Excel
                    excelDataTemperature = excelProcessing.processTemperatureData()
                    timePoints, temperatureData = excelDataTemperature.getData(temperatureFile, testSheetNum = 0)
                    temperatureData = temperatureData*scaleFactor_Temperature # Get Data into micro-Ampes

                    # Process the Data
                    subjectTemperatureFeatures = temperatureAnalysisProtocol.analyzeTemperature(timePoints, temperatureData)
                    
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectTemperatureFeatures) == len(temperatureFeatureNames)
                    
                    # Save the Features and Filtered Data
                    excelProcessingTemperature.saveResults([subjectTemperatureFeatures], temperatureFeatureNames, saveCompiledDataTemperature, "Feature List.xlsx", sheetName = "Temperature Features")

                # Compile the Featues into One Array
                temperatureFeatureLabels.append(featureLabel)
                temperatureFeatures.append(subjectTemperatureFeatures)
                
                if trackRawData:
                    # Read in the temperature Data from Excel
                    excelDataTemperature = excelProcessing.processTemperatureData()
                    timePoints, temperatureData = excelDataTemperature.getData(temperatureFile, testSheetNum = 0)
                    temperatureData = temperatureData*scaleFactor_Temperature # Get Data into micro-Ampes
                    # Track raw feature
                    rawTempData.append([timePoints, temperatureData])  

    # ---------------------- Compile Features Together --------------------- #
    # Compile Labels
    allLabels = []
    if extractGSR: allLabels.append(gsrFeatureLabels) 
    if extractPulse: allLabels.append(pulseFeatureLabels)
    if extractChemical: allLabels.append(chemicalFeatureLabels_ISE)
    if extractChemical: allLabels.append(chemicalFeatureLabels_Enzym)
    if extractTemperature: allLabels.append(temperatureFeatureLabels)
    # Assert That We Have the Same Number of Points in Both
    for labelList in allLabels:
        assert len(labelList) == len(scoreFeatureLabels)
    # Do Not Continue if No Labels Found
    if len(allLabels) == 0:
        sys.exit("Please Specify Features to Extract")
    allLabels = np.array(allLabels)
       
    # Compile Data for Machine Learning
    signalData = []; stressLabels = []; scoreLabels = []
    # Merge the Features
    for arrayInd in range(len(allLabels[0])):
        currentLabels = allLabels[:,arrayInd]
        stressLabel = scoreFeatureLabels[arrayInd]
        
        # If the Subject had All the Data for the Sensor.
        if None not in currentLabels and stressLabel != None:
            # Assert That the Features are the Same
            assert len(set(currentLabels)) == 1
            
            features = []
            # Compile the Features. ORDER MATTERS
            if extractPulse: features.extend(pulseFeatures[arrayInd]) 
            if extractChemical: features.extend(chemicalFeatures_Enzym[arrayInd]) 
            if extractChemical: features.extend(chemicalFeatures_ISE[arrayInd])
            if extractGSR: features.extend(gsrFeatures[arrayInd]) 
            if extractTemperature: features.extend(temperatureFeatures[arrayInd])
            # Save the Compiled Features
            signalData.append(features)
            stressLabels.append(currentLabels[0])
            scoreLabels.append(stressLabel)
    signalData = np.array(signalData)
    stressLabels = np.array(stressLabels)
    scoreLabels = np.array(scoreLabels)
    
    # Compile Feature Names
    featureNames = []        
    featureNames.extend(pulseFeatureNames)
    featureNames.extend(chemicalFeatureNames_Enzym)
    featureNames.extend(chemicalFeatureNames_ISE)
    featureNames.extend(gsrFeatureNames)
    featureNames.extend(temperatureFeatureNames)
    featureNames = np.array(featureNames)
        
    print("Finished Collecting All the Data")
        
    # ----------------------- Extra Feature Analysis ----------------------- #
    print("\nPlotting Feature Comparison")
    
    if plotFeatures:
        if extractChemical:
            chemicalFeatures_ISE = np.array(chemicalFeatures_ISE); chemicalFeatureLabels_ISE = np.array(chemicalFeatureLabels_ISE)
            chemicalFeatures_Enzym = np.array(chemicalFeatures_Enzym); chemicalFeatureLabels_Enzym = np.array(chemicalFeatureLabels_Enzym)
            # Remove None Values
            chemicalFeatures_ISE_NonNone = chemicalFeatures_ISE[chemicalFeatureLabels_ISE != np.array(None)]
            chemicalFeatureLabels_ISE_NonNone = chemicalFeatureLabels_ISE[chemicalFeatureLabels_ISE != np.array(None)]
            chemicalFeatures_Enzym_NonNone = chemicalFeatures_Enzym[chemicalFeatureLabels_Enzym != np.array(None)]
            chemicalFeatureLabels_Enzym_NonNone = chemicalFeatureLabels_Enzym[chemicalFeatureLabels_Enzym != np.array(None)]
            
            # Organize the Features Enzym
            glucoseFeatures = chemicalFeatures_Enzym_NonNone[:, 0:len(glucoseFeatureNames)]
            lactateFeatures = chemicalFeatures_Enzym_NonNone[:, len(glucoseFeatureNames):len(glucoseFeatureNames) + len(lactateFeatureNames)]
            uricAcidFeatures = chemicalFeatures_Enzym_NonNone[:, len(glucoseFeatureNames) + len(lactateFeatureNames):]
            # Organize the Features ISE
            sodiumFeatures = chemicalFeatures_ISE_NonNone[:, 0:len(sodiumFeatureNames)]
            potassiumFeatures = chemicalFeatures_ISE_NonNone[:, len(sodiumFeatureNames):len(sodiumFeatureNames) + len(potassiumFeatureNames)]
            ammoniumFeatures = chemicalFeatures_ISE_NonNone[:, len(sodiumFeatureNames) + len(potassiumFeatureNames):]
                        
            # Plot the Features within a Single Chemical
            analyzeFeatures_ISE = featureAnalysis.featureAnalysis([], [], chemicalFeatureNames_ISE, stimulusTimes_Delayed, dataFolderWithSubjects + "Machine Learning/Compiled Chemical Feature Analysis/")
            analyzeFeatures_ISE.singleFeatureComparison([sodiumFeatures, potassiumFeatures, ammoniumFeatures], [chemicalFeatureLabels_ISE_NonNone, chemicalFeatureLabels_ISE_NonNone, chemicalFeatureLabels_ISE_NonNone], ["Sodium", "Potassium", "Ammonium"], chemicalFeatureNames_ISE)
            analyzeFeatures_Enzym = featureAnalysis.featureAnalysis([], [], chemicalFeatureNames_Enzym, stimulusTimes, dataFolderWithSubjects + "Machine Learning/Compiled Chemical Feature Analysis/")
            analyzeFeatures_Enzym.singleFeatureComparison([glucoseFeatures, lactateFeatures, uricAcidFeatures], [chemicalFeatureLabels_Enzym_NonNone, chemicalFeatureLabels_Enzym_NonNone, chemicalFeatureLabels_Enzym_NonNone], ["Glucose", "Lactate", "UricAcid"], chemicalFeatureNames_Enzym)
            if timePermits:
                # Compare the Features Between Themselves
                analyzeFeatures_Enzym.featureComparison(glucoseFeatures, glucoseFeatures, chemicalFeatureLabels_Enzym_NonNone, glucoseFeatureNames, glucoseFeatureNames, 'Glucose', 'Glucose')
                analyzeFeatures_Enzym.featureComparison(lactateFeatures, lactateFeatures, chemicalFeatureLabels_Enzym_NonNone, lactateFeatureNames, lactateFeatureNames, 'Lactate', 'Lactate')
                analyzeFeatures_Enzym.featureComparison(uricAcidFeatures, uricAcidFeatures, chemicalFeatureLabels_Enzym_NonNone, uricAcidFeatureNames, uricAcidFeatureNames, 'Uric Acid', 'Uric Acid')
                analyzeFeatures_ISE.featureComparison(sodiumFeatures, sodiumFeatures, chemicalFeatureLabels_ISE_NonNone, sodiumFeatureNames, sodiumFeatureNames, 'Sodium', 'Sodium')
                analyzeFeatures_ISE.featureComparison(potassiumFeatures, potassiumFeatures, chemicalFeatureLabels_ISE_NonNone, potassiumFeatureNames, potassiumFeatureNames, 'Potassium', 'Potassium')
                analyzeFeatures_ISE.featureComparison(ammoniumFeatures, ammoniumFeatures, chemicalFeatureLabels_ISE_NonNone, ammoniumFeatureNames, ammoniumFeatureNames, 'Ammonium', 'Ammonium')
                # Cross-Compare the Features Between Each Other
                # analyzeFeatures.featureComparison(lactateFeatures, uricAcidFeatures, chemicalFeatureLabels_NonNone, lactateFeatureNames, uricAcidFeatureNames, 'Lactate', 'Uric Acid')
                # analyzeFeatures.featureComparison(glucoseFeatures, uricAcidFeatures, chemicalFeatureLabels_NonNone, glucoseFeatureNames, uricAcidFeatureNames, 'Glucose', 'Uric Acid')
                # analyzeFeatures.featureComparison(lactateFeatures, glucoseFeatures, chemicalFeatureLabels_NonNone, lactateFeatureNames, glucoseFeatureNames, 'Lactate', 'Glucose')
            
        if extractPulse:
            pulseFeatures = np.array(pulseFeatures)
            pulseFeatureLabels = np.array(pulseFeatureLabels)
            # Remove None Values
            pulseFeatures_NonNone = pulseFeatures[pulseFeatureLabels != np.array(None)]
            pulseFeatureLabels_NonNone = pulseFeatureLabels[pulseFeatureLabels != np.array(None)]
            # Plot the Features within a Pulse
            analyzeFeatures = featureAnalysis.featureAnalysis([], [], pulseFeatureNames, stimulusTimes, dataFolderWithSubjects + "Machine Learning/Compiled Pulse Feature Analysis/")
            analyzeFeatures.singleFeatureComparison([pulseFeatures_NonNone], [pulseFeatureLabels_NonNone], ["Pulse"], pulseFeatureNames)
            if timePermits:
                # Cross-Compare the Features Between Each Other
                analyzeFeatures.featureComparison(pulseFeatures_NonNone, pulseFeatures_NonNone, pulseFeatureLabels_NonNone, pulseFeatureNames, pulseFeatureNames, 'Pulse', 'Pulse')
        
        if extractGSR:
            gsrFeatures = np.array(gsrFeatures)
            gsrFeatureLabels = np.array(gsrFeatureLabels)
            # Remove None Values
            gsrFeatures_NonNone = gsrFeatures[gsrFeatureLabels != np.array(None)]
            gsrFeatureLabels_NonNone = gsrFeatureLabels[gsrFeatureLabels != np.array(None)]
            # Plot the Features within a gsr
            analyzeFeatures = featureAnalysis.featureAnalysis([], [], gsrFeatureNames, stimulusTimes, dataFolderWithSubjects + "Machine Learning/Compiled GSR Feature Analysis/")
            analyzeFeatures.singleFeatureComparison([gsrFeatures_NonNone], [gsrFeatureLabels_NonNone], ["GSR"], gsrFeatureNames)
            if timePermits:
                # Cross-Compare the Features Between Each Other
                analyzeFeatures.featureComparison(gsrFeatures_NonNone, gsrFeatures_NonNone, gsrFeatureLabels_NonNone, gsrFeatureNames, gsrFeatureNames, 'GSR', 'GSR')
                
        if extractTemperature:
            temperatureFeatures = np.array(temperatureFeatures)
            temperatureFeatureLabels = np.array(temperatureFeatureLabels)
            # Remove None Values
            temperatureFeatures_NonNone = temperatureFeatures[temperatureFeatureLabels != np.array(None)]
            temperatureFeatureLabels_NonNone = temperatureFeatureLabels[temperatureFeatureLabels != np.array(None)]
            # Plot the Features within a temperature
            analyzeFeatures = featureAnalysis.featureAnalysis([], [], temperatureFeatureNames, stimulusTimes, dataFolderWithSubjects + "Machine Learning/Compiled Temperature Feature Analysis/")
            analyzeFeatures.singleFeatureComparison([temperatureFeatures_NonNone], [temperatureFeatureLabels_NonNone], ["Temperature"], temperatureFeatureNames)
            if timePermits:
                # Cross-Compare the Features Between Each Other
                analyzeFeatures.featureComparison(temperatureFeatures_NonNone, temperatureFeatures_NonNone, gsrFeatureLabels_NonNone, temperatureFeatureNames, temperatureFeatureNames, 'Temperature', 'Temperature')
                
        #Compare Stress Scores with the Features
        analyzeFeatures = featureAnalysis.featureAnalysis([], [], featureNames, [None, None], dataFolderWithSubjects + "Machine Learning/Compiled Stress Score Feature Analysis/")
        analyzeFeatures.featureComparisonAgainstONE(signalData, scoreLabels, stressLabels, featureNames, "Stress Scores", 'Stress Scores')

    # ---------------------------------------------------------------------- #
    # ---------------------- Machine Learning Analysis --------------------- #
    print("\nBeginning Machine Learning Section")
    
    testStressScores = True
    if testStressScores:
        signalLabels = scoreLabels
        # Machine Learning File/Model Paths + Titles
        modelType = "SVR"  # Machine Learning Options: NN, RF, LR, KNN, SVM, RG, EN, SVR
        supportVectorKernel = "linear" # linear, poly, rbf, sigmoid, precomputed
        modelPath = "./Helper Files/Machine Learning Modules/Models/machineLearningModel_ALL.pkl"
        saveModelFolder = dataFolderWithSubjects + "Machine Learning/" + modelType + "/"
    else:
        signalLabels = stressLabels
        # Machine Learning File/Model Paths + Titles
        modelType = "KNN"  # Machine Learning Options: NN, RF, LR, KNN, SVM, RG, EN, SVR
        supportVectorKernel = "linear" # linear, poly, rbf, sigmoid, precomputed
        modelPath = "./Helper Files/Machine Learning Modules/Models/machineLearningModel_ALL.pkl"
        saveModelFolder = dataFolderWithSubjects + "Machine Learning/" + modelType + "/"
    # Initialize machine learning component.
    performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNames), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)    
    # bestFeatures = featureNames #featureNames_Combinations[np.array(modelScores) >= 0]    
    # newSignalData = performMachineLearning.getSpecificFeatures(featureNames, featureNames, signalData)
    
    
    sys.exit()
            
    # Organize features by type of biomarker
    allChemicalFeatureNames = chemicalFeatureNames_Enzym.copy()
    allChemicalFeatureNames.extend(chemicalFeatureNames_ISE.copy())
    featureNamesListOrder = ["Enzymatic", "ISE", "Chemical", "Pulse"]
    featureNamesList = [chemicalFeatureNames_Enzym, chemicalFeatureNames_ISE, allChemicalFeatureNames, pulseFeatureNames]
    
    # For each biomarker
    for currentFeatureNamesInd in range(len(featureNamesList)):
        featureType = featureNamesListOrder[currentFeatureNamesInd]
        currentFeatureNames = np.array(featureNamesList[currentFeatureNamesInd])
        
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNames), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
        signalData_Good = performMachineLearning.getSpecificFeatures(featureNames, currentFeatureNames, signalData)
        saveFolder = saveModelFolder + featureType + " Feature Combination/"
        saveExcelName = featureType + " Feature Combinations.xlsx"
        
        # numFeaturesCombine = 1
        # performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(currentFeatureNames), machineLearningClasses = listOfStressors, saveDataFolder = saveFolder, supportVectorKernel = supportVectorKernel)
        # modelScores, modelSTDs, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(signalData_Good, signalLabels, currentFeatureNames, numFeaturesCombine, saveData = True, saveExcelName = saveExcelName, printUpdateAfterTrial = 3000000, scaleY = testStressScores)

        # currentFeatureNames = featureNames_Combinations[np.array(modelScores) >= 0.2]    
        # signalData_Good = performMachineLearning.getSpecificFeatures(featureNames, currentFeatureNames, signalData)
        
        numFeaturesCombineList = [5]
        
        # Fit model to all feature combinations
        for numFeaturesCombine in numFeaturesCombineList:
            print(saveExcelName, numFeaturesCombine)
            performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(currentFeatureNames), machineLearningClasses = listOfStressors, saveDataFolder = saveFolder, supportVectorKernel = supportVectorKernel)
            modelScores, modelSTDs, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(signalData_Good, signalLabels, currentFeatureNames, numFeaturesCombine, saveData = True, saveExcelName = saveExcelName, printUpdateAfterTrial = 3000000, scaleY = testStressScores)
       
                
    # numFeaturesCombine = 1
    # saveExcelName = "All Feature Combinations Above R2 20.xlsx"
    # performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNames), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
    # modelScores, modelSTDs, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(signalData, signalLabels, featureNames, numFeaturesCombine, saveData = False, saveExcelName = saveExcelName, printUpdateAfterTrial = 1000000, scaleY = testStressScores)

    # bestFeatures = featureNames_Combinations[np.array(modelScores) >= 0.20]    
    # newSignalData = performMachineLearning.getSpecificFeatures(featureNames, bestFeatures, signalData)
                
    # # Fit model to all feature combinations
    # # saveExcelName = "All Feature Combinations.xlsx"
    # for numFeaturesCombine in [5]:
    #     print(saveExcelName, numFeaturesCombine)
    #     performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(bestFeatures), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
    #     modelScores, modelSTDs, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(newSignalData, signalLabels, bestFeatures, numFeaturesCombine, saveData = True, saveExcelName = saveExcelName, printUpdateAfterTrial = 1000000, scaleY = testStressScores)


    sys.exit("STOPPING")
    
    # modelScores_Single0 = []
    # modelScores_Single1 = []
    # modelScores_Single2 = []
    # modelScores_SingleTotal = []
    # for featureInd in range(len(featureNames)):
    #     featureRow = featureNames[featureInd]
    
    #     signalDataCull = np.reshape(signalData[:,featureInd], (-1,1))
    
    #     performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = 1, machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
        
    #     modelScore = performMachineLearning.scoreClassificationModel(signalDataCull, signalLabels, stratifyBy = stressLabels)
        
    #     # modelScores_Single0.append(modelScore[0])
    #     # modelScores_Single1.append(modelScore[1])
    #     # modelScores_Single2.append(modelScore[2])
    #     # modelScores_SingleTotal.append(modelScore[3])
        
    #     modelScores_SingleTotal.append(modelScore[0])
        
    # # excelProcessing.processMLData().saveFeatureComparison([modelScores_Single0], [], featureNames, saveModelFolder, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison Cold")
    # # excelProcessing.processMLData().saveFeatureComparison([modelScores_Single1], [], featureNames, saveModelFolder, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison Excersize")
    # # excelProcessing.processMLData().saveFeatureComparison([modelScores_Single2], [], featureNames, saveModelFolder, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison VR")
    # excelProcessing.processMLData().saveFeatureComparison([modelScores_SingleTotal], [], featureNames, saveModelFolder, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison Total")
    
    # modelScores = np.zeros((len(featureNames), len(featureNames)))
    # for featureIndRow in range(len(featureNames)):
    #     print(featureIndRow)
    #     featureRow = featureNames[featureIndRow]
    #     for featureIndCol in range(len(featureNames)):
    #         if featureIndCol < featureIndRow:
    #              modelScores[featureIndRow][featureIndCol] = modelScores[featureIndCol][featureIndRow]
    #              continue
             
    #         featureCol = featureNames[featureIndCol]
    #         signalDataCull = np.dstack((signalData[:,featureIndRow], signalData[:,featureIndCol]))[0]
            
    #         performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = 2, machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
    #         modelScore = performMachineLearning.trainModel(signalDataCull, signalLabels, returnScore = True, stratifyBy = stressLabels)
    #         modelScores[featureIndRow][featureIndCol] = modelScore
    # for featureIndRow in range(len(featureNames)):
    #     featureRow = featureNames[featureIndRow]
    #     for featureIndCol in range(len(featureNames)):
    #         if featureIndCol < featureIndRow:
    #              modelScores[featureIndRow][featureIndCol] = modelScores[featureIndCol][featureIndRow]
    #              continue
    # excelProcessing.processMLData().saveFeatureComparison(modelScores, featureNames, featureNames, saveModelFolder, "Pairwise Feature Accuracy.xlsx", sheetName = "Feature Comparison")

    sys.exit()
    

    


    bestFeatureNames = featureNames_Combinations[np.array(modelScores) >= 0.9]
    featureFound, featureFoundCounter = performMachineLearning.countScoredFeatures(bestFeatureNames)
    
    from scipy import stats
    stats.trim_mean(featureFoundCounter, 0.2)
    plt.plot(featureFoundCounter, 'o')
    len(featureFound)
    
    minCount = stats.trim_mean(featureFoundCounter, 0.2)
    # minCount = 2
    plt.bar(featureFound[featureFoundCounter > minCount], featureFoundCounter[featureFoundCounter > minCount])
    plt.xticks(rotation='vertical')
    
    
    minCount = 2

    featureNamesPermute_Good = featureFound[featureFoundCounter > minCount]
    signalData_Good_New = performMachineLearning.getSpecificFeatures(featureNames, featureNamesPermute_Good, signalData)


    numFeaturesCombine = 5
    performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNamesPermute_Good), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder + "_CullFrom3/", supportVectorKernel = supportVectorKernel)
    modelScores, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(signalData_Good, signalLabels, featureNamesPermute_Good, numFeaturesCombine, saveData = True, printUpdateAfterTrial = 15000, scaleY = testStressScores)
        
    
    
    
    
    
    bestFeatures = ['peripheralAugmentationIndex_EST_SignalIncrease', 'systolicUpSlopeTime_StressLevel', 'tidalPeakTime_StressLevel', 'reflectionIndex_SignalIncrease', 'dicroticNotchToTidalDuration_SignalIncrease']        
    bestFeatures = []
    bestFeatures.extend(pulseFeatureNames)
    bestFeatures.extend(chemicalFeatureNames_Enzym)
    bestFeatures.extend(chemicalFeatureNames_ISE)
    bestFeatures.extend(gsrFeatureNames)
    bestFeatures.extend(temperatureFeatureNames)
    
    bestFeatures = np.array(bestFeatures)
    signalData_Good = performMachineLearning.getSpecificFeatures(featureNames, bestFeatures, signalData)
        
    saveFolder = saveModelFolder + "_AllFeatures/"
    
    numFeaturesCombine = 1
    performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(bestFeatures), machineLearningClasses = listOfStressors, saveDataFolder = saveFolder, supportVectorKernel = supportVectorKernel)
    modelScores, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(signalData_Good, signalLabels, bestFeatures, numFeaturesCombine, saveData = False, printUpdateAfterTrial = 15000, scaleY = testStressScores)

    featureNamesPermute_Good = featureNames_Combinations[np.array(modelScores) >= 0.48]    
    signalData_Good = performMachineLearning.getSpecificFeatures(featureNames, featureNamesPermute_Good, signalData)
    
    for numFeaturesCombine in [5,6,7,8]:
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNamesPermute_Good), machineLearningClasses = listOfStressors, saveDataFolder = saveFolder, supportVectorKernel = supportVectorKernel)
        modelScores, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(signalData_Good, signalLabels, featureNamesPermute_Good, numFeaturesCombine, saveData = True, printUpdateAfterTrial = 100000, scaleY = testStressScores)
    

    
    sys.exit()
    
    
    if False:
        bestFeatures = ['systolicUpstrokeAccelMinVel_StressLevel', 'systolicUpSlopeArea_SignalIncrease', 'velDiffConc_Glucose', 'accelDiffMaxConc_Glucose', 'rightDiffAmp_Lactate']
        
        bestFeatures = ['reflectionIndex_SignalIncrease', 'dicroticPeakVel_SignalIncrease', 'dicroticPeakAccel_SignalIncrease', 'peripheralAugmentationIndex_EST_SignalIncrease']
        bestFeatures.extend(['bestprominence_GSR'])
        
        #bestFeatures = ['meanStressIncrease_Ammonium', 'dicroticFallVelMinVel_StressLevel', 'endSlope_StressLevel']
        #bestFeatures.extend(['bestprominence_GSR'])
        bestFeatures.extend(['endSlope_StressLevel', 'dicroticFallVelMinVel_StressLevel'])
        
        bestFeatures = ['peripheralAugmentationIndex_EST_SignalIncrease', 'systolicUpSlopeTime_StressLevel', 'tidalPeakTime_StressLevel', 'meanSignalRecovery_GSR', 'stressSlope_Ammonium']

        newSignalData = performMachineLearning.getSpecificFeatures(featureNames, bestFeatures, signalData)
        
        bestFeatures = ['pAIx_Increase', 'systolicRiseDuration', 'tidalPeakTime_StressLevel', 'meanSignalRecovery_GSR', 'stressSlope_Ammonium']       
        
        bestFeatures = ['velToVelArea_Glucose' ,'peakSTD_GSR', 'systolicRiseDuration_SignalIncrease', 'meanStressIncrease_Temperature']

    if False:
        bestFeatures = ['pulseDuration_SignalIncrease', 'stressSlope_Sodium', 'prominenceRatio_GSR', 'systolicDicroticNotchAmpRatio_SignalIncrease', 'peakSTD_Ammonium', 'maxUpSlopeConc_Glucose']
        newSignalData = performMachineLearning.getSpecificFeatures(featureNames, bestFeatures, signalData)
        
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(bestFeatures), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
        performMachineLearning.accuracyDistributionPlot_Average(signalData_Standard, signalLabels, listOfStressors, analyzeType = "Full", name = "Accuracy Distribution", testSplitRatio = 0.3)


    sys.exit()
    
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #

    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    
    bestFeatures = ['velToVelArea_Glucose' ,'peakSTD_GSR', 'systolicRiseDuration_SignalIncrease', 'meanStressIncrease_Temperature']
    newSignalData = performMachineLearning.getSpecificFeatures(featureNames, bestFeatures, signalData)
    signalData_Standard = sc_X.fit_transform(newSignalData)
    
    # Header = ['Stress Score', 'Stress Class']
    # Header.extend(bestFeatures)
    # saveInfo = np.hstack((stressLabels.reshape(-1,1), newSignalData))
    # saveInfo = np.hstack((signalLabels.reshape(-1,1), saveInfo))
    # excelProcessing.dataProcessing().saveResults(saveInfo, Header, './', 'Final Features.xlsx')

    performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(bestFeatures), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
    
    if testStressScores:
        sc_y = StandardScaler()
        signalLabels_Standard = sc_y.fit_transform(signalLabels.reshape(-1, 1))
        performMachineLearning.predictionModel.trainModel(signalData_Standard, signalLabels_Standard, signalData_Standard, signalLabels_Standard)
        performMachineLearning.predictionModel.scoreModel(signalData_Standard, signalLabels_Standard)
    else:
        performMachineLearning.predictionModel.trainModel(signalData_Standard, signalLabels, signalData_Standard, signalLabels)
        performMachineLearning.predictionModel.scoreModel(signalData_Standard, signalLabels)
        
    if False:
        from sklearn.model_selection import train_test_split
        
        sc_y = StandardScaler()
        signalLabels_Standard = sc_y.fit_transform(signalLabels.reshape(-1, 1))
        
        modelScores = []
        for _ in range(10000):
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData_Standard, signalLabels_Standard, test_size=0.2, shuffle= True, stratify=stressLabels)
    
            modelScores.append(performMachineLearning.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels))
        
        # Display the Spread of Scores
        plt.hist(modelScores, 50, facecolor='blue', alpha=0.5)
        # Fit the Mean Distribution and Save the Mean
        ae, loce, scalee = stats.skewnorm.fit(modelScores)
        # Take the Median Score as the True Score
        meanScore = np.round(loce*100, 2)

    

    # Get the fit parameters
    intercept = performMachineLearning.predictionModel.model.intercept_[0]
    featureCoefficients = performMachineLearning.predictionModel.model.coef_[0]
    # Initialize 
    predictedValues = 0
    stressEquation = ""
    offset = intercept*sc_y.scale_[0] + sc_y.mean_[0]
    
    # For each feature
    for featureNum in range(len(featureCoefficients)):
        # Get the feature information
        featureCoef = featureCoefficients[featureNum] # The feature coefficient for the scaled data
        featureName = bestFeatures[featureNum] # The current feature name
        
        # Calculate the offset and scale factors
        scale = sc_y.scale_[0] * featureCoef / sc_X.scale_[featureNum]
        offset -= sc_y.scale_[0] * featureCoef * sc_X.mean_[featureNum] / sc_X.scale_[featureNum]
        
        # Track the equation
        stressEquation += "{:e}".format(scale) + "*" + featureName + " + "
        # Predict the a data point
        predictedValues += scale*newSignalData[0][featureNum]
        
    predictedValues += offset
    stressEquation += "{:e}".format(offset) 
    print(stressEquation, "\nPrediceted Value: ", predictedValues)
    
    # 8.697465e-04*np.max(abs(newSignalData[:,0])), 7.701087e+00*np.max(abs(newSignalData[:,1])), -1.319248e+02*np.max(abs(newSignalData[:,2])), 1.775774e+01*np.max(abs(newSignalData[:,3]))
    
    shap_values = performMachineLearning.featureImportance(signalData_Standard, signalLabels_Standard, signalData_Standard, signalLabels_Standard, featureLabels = bestFeatures, numTrials = 1)

    import shap
    import pandas as pd
    
    featureLabels = np.array(bestFeatures)
    print("Entering SHAP Analysis")
    # Create Panda DataFrame to Match Input Type for SHAP
    testingDataPD = pd.DataFrame(signalData_Standard, columns = featureLabels)
    testingDataPD_Full = pd.DataFrame(newSignalData, columns = featureLabels)
    
    # More General Explainer
    explainerGeneral = shap.Explainer(performMachineLearning.predictionModel.model.predict, testingDataPD)
    shap_valuesGeneral = explainerGeneral(testingDataPD)
    
    # Calculate Shap Values
    explainer = shap.KernelExplainer(performMachineLearning.predictionModel.model.predict, testingDataPD)
    shap_values = explainer.shap_values(testingDataPD, nsamples=len(signalData_Standard))
    
    shap_values = sc_y.inverse_transform(shap_values) - np.mean(sc_y.inverse_transform(shap_values), axis=0)
    shap_valuesGeneral.values =  shap_values
     
    # Specify Indivisual Sharp Parameters
    dataPoint = 3
    featurePoint = 2
    explainer.expected_value = sc_y.inverse_transform([[explainer.expected_value]])[0][0]
    shap_valuesGeneral.base_values = sc_y.inverse_transform([shap_valuesGeneral.base_values])[0]
    
    
    # Summary Plot
    name = "Summary Plot"
    summaryPlot = plt.figure()
    shap.summary_plot(shap_valuesGeneral, testingDataPD_Full, feature_names = featureLabels)
    summaryPlot.savefig(performMachineLearning.saveDataFolder + "SHAP Values/" + name + " " + performMachineLearning.modelType + ".png", bbox_inches='tight', dpi=300)
            
    
    # Decision Plot
    name = "Decision Plot"
    decisionPlotOne = plt.figure()
    shap.decision_plot(explainer.expected_value, shap_values, features = testingDataPD_Full, feature_names = featureLabels, feature_order = "importance")
    decisionPlotOne.savefig(performMachineLearning.saveDataFolder + "SHAP Values/" + name + " " + performMachineLearning.modelType + ".png", bbox_inches='tight', dpi=300)
            
    # Bar Plot
    name = "Bar Plot"
    barPlot = plt.figure()
    shap.plots.bar(shap_valuesGeneral, max_display = len(featureLabels), show = True)
    barPlot.savefig(performMachineLearning.saveDataFolder + "SHAP Values/" + name + " " + performMachineLearning.modelType + ".png", bbox_inches='tight', dpi=300)

    name = "Segmented Bar Plot"
    barPlotSegmeneted = plt.figure()
    labelTypes = [listOfStressors[ind] for ind in stressLabels]
    shap.plots.bar(shap_valuesGeneral.cohorts(labelTypes).abs.mean(0))
    barPlotSegmeneted.axes[0].legend(loc="lower right")
    barPlotSegmeneted.savefig(performMachineLearning.saveDataFolder + "SHAP Values/" + name + " " + performMachineLearning.modelType + "_Segmented.png", bbox_inches='tight', dpi=300)

    # HeatMap Plot
    name = "Heatmap Plot"
    heatmapPlot = plt.figure()
    shap.plots.heatmap(shap_valuesGeneral, max_display = len(featureLabels), show = True, instance_order=shap_valuesGeneral.sum(1))
    heatmapPlot.savefig(performMachineLearning.saveDataFolder + "SHAP Values/" + name + " " + performMachineLearning.modelType + ".png", bbox_inches='tight', dpi=300)
                    
    
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #    
    
    
    predictedLabels = performMachineLearning.predictionModel.predictData(signalData_Standard)
    
    signalLabels_Unscaled = sc_y.inverse_transform(signalLabels_Standard).reshape(1,-1)[0]
    predictedLabels_Unscaled = sc_y.inverse_transform(predictedLabels.reshape(-1, 1)).reshape(1,-1)[0]
    
    fitParams = np.polyfit(signalLabels_Unscaled.flat, predictedLabels_Unscaled.flat, 1)
    p = np.poly1d(fitParams)
    
    plt.plot(sorted(signalLabels_Unscaled), p(sorted(signalLabels_Unscaled)), '--', c='tab:red')
    plt.fill_between(sorted(signalLabels_Unscaled), p(sorted(signalLabels_Unscaled))-2, 2+p(sorted(signalLabels_Unscaled)), color="#ABABAB", label='Uncertainty in Stress Score')
    plt.plot(sorted(signalLabels_Unscaled), 2+p(sorted(signalLabels_Unscaled)), '-', c='tab:blue')
    plt.plot(sorted(signalLabels_Unscaled), p(sorted(signalLabels_Unscaled))-2, '-', c='tab:blue')
    plt.plot(signalLabels_Unscaled, predictedLabels_Unscaled, 'o', c='black')
    plt.title("Stress Prediction Accuracy")
    plt.ylabel("Predicted Stress Score")
    plt.xlabel("Actual Stress Score")
    plt.legend(['$R^2$ = 0.8769', 'Uncertainty in Score'])
    plt.show()
    
    
    plt.hlines(3.8, min(signalLabels_Unscaled), max(signalLabels_Unscaled), 'k')
    plt.plot(signalLabels_Unscaled, abs(signalLabels_Unscaled - predictedLabels_Unscaled), 'o', c='tab:purple')
    plt.hlines(3, min(signalLabels_Unscaled), max(signalLabels_Unscaled), 'k')
    plt.title("Stress Prediction Error")
    plt.ylabel("Error in Stress Score Prediction")
    plt.xlabel("Actual Stress Score")
    
    
    fitParams = np.polyfit(x, y, 5)
    p = np.poly1d(fitParams)
    
    
    # xAllSensors = [0, 1, 2, 3, 4, 5, 6]
    # yAllSensors = [0, 0.544, 0.731, 0.792, 0.8549, 0.8828, 0.917]
    
    # xChemical = [0, 1, 2, 3, 4, 5, 6][0:-1]
    # yChemical = [0, 0.17859, 0.327, 0.47205, 0.5788, 0.6485]
    
    # xChemicalISE = [0, 1, 2, 3, 4, 5, 6]
    # yChemicalISE = [0, 0.167, 0.3018, 0.3336, 0.3448, 0.3771, 0.3922]
    
    # xChemicalEnzym = [0, 1, 2, 3, 4, 5, 6][0:-1]
    # yChemicalEnzym = [0, 0.17859, 0.32735, 0.44282, 0.54868, 0.59554]
    
    # xPulse = [0, 1, 2, 3, 4, 5, 6][0:-2]
    # yPulse = [0, 0.544, 0.731, 0.7724, 0.7875]
    
    # xGSR = [0, 1, 2, 3, 4, 5, 6]
    # yGSR = [0, 0.3415, 0.3615, 0.3949, 0.3999, 0.3995, 0.3969]
    
    # xTemp = [0, 1, 2, 3, 4, 5, 6]
    # yTemp = [0, 0.167, 0.2898, 0.3437, 0.3488, 0.3452, 0.302]
    
    xAllSensors = [0, 1, 2, 3, 4, 5]
    yAllSensors = [0, 0.754792733, 0.793598521, 0.843489319, 0.876935824, 0.8983380733751253]
    
    xChemical = [0, 1, 2, 3, 4, 5]
    yChemical = [0, 0.754792733, 0.767105711, 0.768359606, 0.775994562, 0.779629842]
    
    xPulse = [0, 1, 2, 3, 4, 5]
    yPulse = [0, 0.4679258, 0.67592309, 0.715226522, 0.778745543, 0.809167892]
    
    xGSR = [0, 1, 2, 3, 4, 5]
    yGSR = [0, 0.565892316, 0.734183539, 0.732179347, 0.725954197, 0.724841668]
    
    xTemp = [0, 1, 2, 3, 4, 5]
    yTemp = [0, 0.332047234, 0.562707762, 0.581873771, 0.581341113, 0.581626611]

    plt.plot(xAllSensors, yAllSensors, 'o--', c='black', markersize=4, label = "All Features")
    # plt.plot(xChemicalEnzym, yChemicalEnzym, 'o--', c='tab:blue', markersize=4, label = "Enzymatic Features")
    # plt.plot(xChemicalISE, yChemicalISE, 'o--', c='tab:blue', markersize=4, label = "ISE Features")

    plt.plot(xPulse, yPulse, 'o--', c='tab:blue', markersize=4, label = "Pulse Features")
    plt.plot(xGSR, yGSR, 'o--', c='tab:green', markersize=4, label = "GSR Features")
    plt.plot(xChemical, yChemical, 'o--', c='tab:brown', markersize=4, label = "Chemical Features")
    plt.plot(xTemp, yTemp, 'o--', c='tab:red', markersize=4, label = "Temperature Features")

    plt.title("Accuracy Convergence")
    plt.xlabel("Number of Features")
    plt.ylabel("$R^2$ Score")
    legend = plt.legend()
    plt.ylim(-.05,1)
    plt.savefig(saveModelFolder + "accuracyConvergence.png", dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')

        
    
    

    
    
    
    fig, ax1 = plt.subplots()
    plt.title("Stress Prediction Accuracy")
    
    ax1.plot(signalLabels_Unscaled, predictedLabels_Unscaled, 'o', color='tab:brown')
    ax1.set_xlabel("Actual Stress Score")
    ax1.set_ylabel("Predicted Stress Score")
    ax1.tick_params(axis='y', labelcolor='tab:brown')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Prediction Error', color=color)  # we already handled the x-label with ax1
    ax2.plot(signalLabels_Unscaled, abs(predictedLabels_Unscaled - signalLabels_Unscaled), 'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 30)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    
    
    
    
    
    
    
    
    bestFeatures = ['centralAugmentationIndex_EST_SignalIncrease', 'systolicUpSlopeTime_StressLevel', 'tidalPeakTime_StressLevel', 'reflectionIndex_SignalIncrease', 'dicroticNotchTime_StressLevel']        
    # bestFeatures = []
    # bestFeatures.extend(pulseFeatureNames)
    # bestFeatures.extend(chemicalFeatureNames_Enzym)
    # bestFeatures.extend(chemicalFeatureNames_ISE)
    # bestFeatures.extend(gsrFeatureNames)
    bestFeatures.extend(temperatureFeatureNames)
    
    newSignalData = performMachineLearning.getSpecificFeatures(featureNames, bestFeatures, signalData)
    
    numFeaturesCombine = 1
    performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(bestFeatures), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
    modelScores, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(newSignalData, signalLabels, bestFeatures, numFeaturesCombine, saveData = False, printUpdateAfterTrial = 15000, scaleY = testStressScores)

    bestFeatures = featureNames_Combinations[np.array(modelScores) >= 0]    
    newSignalData = performMachineLearning.getSpecificFeatures(featureNames, bestFeatures, signalData)
    
    numFeaturesCombine = 3
    performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(bestFeatures), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
    modelScores, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(newSignalData, signalLabels, bestFeatures, numFeaturesCombine, saveData = False, printUpdateAfterTrial = 15000, scaleY = testStressScores)

    import matplotlib.pyplot as plt

    labels = ['Pulse + feature', '2 Pulse + feature', 'Pulse + 2 features', '5 features']
    # Add contirbutions
    pulseContributions = [
        [[0.486, 0.526, 0.486, 0.544], [0, 0.205, 0, 0.187], [0, 0, 0, 0.0414], [0, 0, 0, 0.0151], [0, 0, 0, 0.01]], # Pulse with GSR
        [[0.486, 0.526, 0.486, 0], [0, 0.192, 0, 0]], # Pulse wth chemical
        [[0.526, 0.526, 0.486, 0], [0, 0.192, 0, 0]], # Pulse with temp
        [[0.544, 0.526, 0.544, 0], [0, 0.205, 0, 0]], # Pulse by itself
    ]
    chemicalContributions = [
        [[0, 0, 0, 0]], # Pulse with GSR
        [[0.1635, 0.0742, 0.1635, 0.17859], [0, 0, 0.0728, 0.1484],[0, 0, 0, 0.14505], [0, 0, 0, .10675], [0, 0, 0, 0.0697]], # Pulse wth chemical
        [[0, 0, 0, 0]],# Pulse with temp
        [[0, 0, 0, 0]], # Pulse by itself
    ]
    gsrContributions = [
        [[0.203, 0.0308, 0.203, 0], [0, 0, 0.0533, 0]], # Pulse with GSR
        [[0, 0, 0, 0]], # Pulse wth chemical
        [[0, 0, 0, 0.3415], [0, 0, 0, 0.02],[0, 0, 0, 0.0334], [0, 0, 0, .005], [0, 0, 0, 0]],# Pulse with temp
        [[0, 0, 0, 0]], # Pulse by itself
    ]
    temperatureContributions = [
        [[0, 0, 0, 0]], # Pulse with GSR
        [[0, 0, 0, 0]], # Pulse wth chemical
        [[0.0587, 0.0413, 0.0759, 0], [0, 0, 0.0532, 0]], # Pulse with temp
        [[0, 0, 0, 0.167], [0, 0, 0, 0.1228],[0, 0, 0, 0.0539], [0, 0, 0, .0051], [0, 0, 0, 0]], # Pulse by itself
    ]
    # Specify plot aesthetics
    totalLength = 1
    width = totalLength/(1+len(pulseContributions))       # the width of the bars: can also be len(x) sequence
                
    fig, ax = plt.subplots()
    
    x_axis = np.arange(len(labels))*totalLength
    for index in range(len(pulseContributions)):
        pulseValues = np.array(pulseContributions[index])
        chemicalValues = np.array(chemicalContributions[index])
        gsrValues = np.array(gsrContributions[index])
        tempValues = np.array(temperatureContributions[index])
        
        for ind, pulseValue in enumerate(pulseValues):
            ax1 = ax.bar(x_axis + width*(index+totalLength/2), pulseValue, width, bottom=sum(pulseValues[0:ind]), color="royalblue", edgecolor = "black")
        for ind, chemicalValue in enumerate(chemicalValues):
            ax2 = ax.bar(x_axis + width*(index+totalLength/2), chemicalValue, width, bottom=sum(chemicalValues[0:ind])+sum(pulseValues), color="tab:purple", edgecolor = "black")
        for ind, gsrValue in enumerate(gsrValues):
            ax3 = ax.bar(x_axis + width*(index+totalLength/2), gsrValue, width, bottom=sum(gsrValues[0:ind])+sum(pulseValues)+sum(chemicalValues), color="mediumseagreen", edgecolor = "black")
        for ind, tempValue in enumerate(tempValues):
            ax4 = ax.bar(x_axis + width*(index+totalLength/2), tempValue, width, bottom=sum(tempValues[0:ind])+sum(pulseValues)+sum(chemicalValues)+sum(gsrValues), color="tomato", edgecolor = "black")
    
    ax.set_ylabel('$R^2$ Contributions')
    ax.set_title('$R^2$ Contributions from each Biomarker')
    a = plt.xticks(x_axis+(totalLength-width)/2, labels, fontsize=8)
    plt.ylim(0,1)
    ax.legend([ax1, ax2, ax3, ax4], ['$Pulse$', '$Chemical$', '$GSR$', '$Temperature$'], loc = 'center', bbox_to_anchor=(1.18, .835))
    
    plt.show()


    
    
    import plotapi
    from plotapi import Chord
    
    import pandas as pd
    import holoviews as hv
    from holoviews import opts, dim
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    
    if False:
        bestFeatures = ['peripheralAugmentationIndex_EST_SignalIncrease', 'systolicUpSlopeTime_StressLevel', 'tidalPeakTime_StressLevel', 'reflectionIndex_SignalIncrease', 'dicroticNotchToTidalDuration_SignalIncrease']        
        bestFeatures.extend(chemicalFeatureNames_Enzym)
        bestFeatures.extend(chemicalFeatureNames_ISE)
        bestFeatures.extend(gsrFeatureNames)
        bestFeatures.extend(temperatureFeatureNames)
        signalData_Good = performMachineLearning.getSpecificFeatures(featureNames, bestFeatures, signalData)
        
        numFeaturesCombine = 2
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(bestFeatures), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
        modelScores, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(signalData_Good, signalLabels, bestFeatures, numFeaturesCombine, saveData = False, printUpdateAfterTrial = 15000, scaleY = testStressScores)
    else:
        numFeaturesCombine = 2
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNames), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
        modelScores, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(signalData, signalLabels, featureNames, numFeaturesCombine, saveData = False, printUpdateAfterTrial = 15000, scaleY = testStressScores)
   
    
    featureNamesPermute_Good = featureNames_Combinations[np.array(modelScores) >= 0.54]
    
    selectedFeatures = ['Pulse', 'Uric Acid', 'Glucose', 'Lactate', 'GSR', 'Ammonium', 'Temperature', 'Sodium', 'Potassium']
    selectedFeaturesTypes = ['StressLevel', 'UricAcid', 'Glucose', 'Lactate', 'GSR', 'Ammonium', 'Temperature', 'Sodium', 'Potassium']
    def findType(name):
        try:
            index = selectedFeaturesTypes.index(name)
        except:
            if name == 'SignalIncrease':
                index = 0
            else:
                print("AHHHHHH: ", name)
        return index
    
    featureLists = []
    bestFeatures.append(pulseFeatureNames)
    bestFeatures.append(chemicalFeatureNames_Enzym)
    bestFeatures.append(chemicalFeatureNames_ISE)
    bestFeatures.append(gsrFeatureNames)
    bestFeatures.append(temperatureFeatureNames)
    
    featureOrganization = [[] for _ in range(len(selectedFeatures))]
    for feature in featureNames:
        featureOrganization[findType(feature.split('_')[-1])].append(feature)
    
    bestFeatures_inEach = []
    for featureInd in range(len(featureOrganization)):
        currentFeatures = featureOrganization[featureInd]
        signalData_ofType = performMachineLearning.getSpecificFeatures(featureNames, currentFeatures, signalData)
        
        numFeaturesCombine = 1
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(currentFeatures), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
        modelScores, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(signalData_ofType, signalLabels, currentFeatures, numFeaturesCombine, saveData = False, printUpdateAfterTrial = 15000, scaleY = testStressScores)

        bestFeatures_inEach.extend(featureNames_Combinations[0:3])
    bestData_inEach = performMachineLearning.getSpecificFeatures(featureNames, bestFeatures_inEach, signalData)
    
    sc_X = StandardScaler()
    signalData_Standard = sc_X.fit_transform(bestData_inEach)
    matrix = np.array(np.corrcoef(signalData_Standard.T)); 
    
    correlationValues = np.zeros((len(selectedFeatures), len(selectedFeatures)))
    counterValues = np.zeros((len(selectedFeatures), len(selectedFeatures)))
    for sourceInd in range(len(matrix)):
        for targetInd in range(sourceInd+1, len(matrix)): 
            sourceName = bestFeatures_inEach[sourceInd]
            targetName = bestFeatures_inEach[targetInd]
            
            sourceNameInd = findType(sourceName.split('_')[-1])
            targetNameInd = findType(targetName.split('_')[-1])
                        
            corVal = abs(matrix[sourceInd][targetInd])
            correlationValues[sourceNameInd][targetNameInd] += corVal
            correlationValues[targetNameInd][sourceNameInd] += corVal
            
            counterValues[sourceNameInd][targetNameInd] += 1
            counterValues[targetNameInd][sourceNameInd] += 1
    correlationValues = list(np.round(correlationValues/counterValues, 4)*100)
    correlationValues = [list(np.round(i)) for i in correlationValues]
    
    for i in range(len(correlationValues)):
        correlationValues[i][i] = 0


    import plotapi
    from sklearn.preprocessing import StandardScaler
    from plotapi import Chord
    import scipy
    
    plotapi.api_key("ff69182a-324b-46b9-90c5-77a6da9bd050")
    Chord.api_key("ff69182a-324b-46b9-90c5-77a6da9bd050")

    selectedFeatures = ['Pulse', 'GSR', 'Glucose', 'Lactate', 'Uric Acid', 'Sodium', 'Potassium', 'Ammonium', 'Temperature']
    rawFeatureList = [[] for _ in range(len(selectedFeatures))]
    rawDataList = [rawPulseFeatureData, rawGSRData, rawEnzymData, rawISEData, rawTempData]
    newTimes = np.arange(100, 1900, 1)
    timeOffsets = [-1, -1, 0, 0, 0, 1, 1, 1, -1]
    
    
    index = 0
    # interpolate the raw data
    for rawFeaturesInd in range(len(rawDataList)):
        rawFeatures = rawDataList[rawFeaturesInd]
        timeOffset = timeOffsets[rawFeaturesInd]
        
        for rawFeature in rawFeatures:
            rawFeature = np.array(rawFeature)
            x = rawFeature[0]
            
            for yInd in range(len(rawFeature[1:])):
                y = rawFeature[1 + yInd]
                
                f = scipy.interpolate.interp1d(x, y, kind='linear') 
                if timeOffset == 1:
                    newData = f(newTimes + 500)
                else:
                    newData = f(newTimes)
                newData = (np.array(newData) - np.mean(newData))/np.std(newData)
                rawFeatureList[index + yInd].append(newData)
        index = index + 1 + yInd
    rawFeatureList = np.array(rawFeatureList)
    # Scale the data
    # for dataInd in range(len(rawFeatureList)):
    #     sc_X = StandardScaler()
    #     rawFeatureList[dataInd] = sc_X.fit_transform(rawFeatureList[dataInd])
    
    correlationMatrixFull = np.zeros((len(selectedFeatures)*2, 2*len(selectedFeatures)))
    correlationList = [[] for _ in range(len(selectedFeatures))]
    for rawFeatureInd in range(len(rawFeatureList[0])):
        
        currentFeatures = rawFeatureList[:,rawFeatureInd]
        correlationMatrixFull += abs(np.corrcoef(currentFeatures, currentFeatures))
    correlationMatrixFull = correlationMatrixFull/len(rawFeatureList[0])
    
    correlationMatrix = correlationMatrixFull[0:len(selectedFeatures), 0:len(selectedFeatures)]
    
    correlationMatrix = list(np.round(correlationMatrix, 4)*100)
    correlationMatrix = [list(np.round(i)) for i in correlationMatrix]
    for i in range(len(correlationMatrix)):
        correlationMatrix[i][i] = 0

    print(correlationMatrix)
    
    Chord(
        correlationMatrix,
        selectedFeatures,
        margin=150,
        thumbs_margin=1,
        popup_width=1000,
        directed=False,
        arc_numbers=False,
        animated_intro=True,
        noun="percent correlated",
        data_table_show_indices=False,
        title="Average Feature Correlation Across Biomarkers",
    ).to_html()
    
    
    
    from plotapi import Sankey
    
    selectedFeatures = ['Pulse', 'GSR', 'Glucose', 'Lactate', 'Uric Acid', 'Ammonium', 'Sodium', 'Potassium', 'Temperature']
    
    selectedFeaturesTypes = ['StressLevel', 'GSR', 'Glucose', 'Lactate', 'UricAcid', 'Ammonium', 'Sodium', 'Potassium', 'Temperature']
    def findType(name):
        try:
            index = selectedFeaturesTypes.index(name)
        except:
            if name == 'SignalIncrease':
                index = 0
            else:
                print("AHHHHHH: ", name)
        return index
    
    counterValues = np.zeros((len(selectedFeatures), len(selectedFeatures)))
    for sourceInd in range(len(matrix)):
        for targetInd in range(sourceInd+1, len(matrix[0])): 
            sourceName = featureNamesPermute_Good[sourceInd]
            targetName = featureNamesPermute_Good[targetInd]
            
            sourceNameInd = findType(sourceName.split('_')[-1])
            targetNameInd = findType(targetName.split('_')[-1])
            
            counterValues[sourceNameInd][targetNameInd] += 1
            counterValues[targetNameInd][sourceNameInd] += 1
    
    storeSankeyInfo = []
    for featureInd in range(1, len(counterValues[0])):        
        sourceNameInd = featureInd
        targetNameInd = 0
        
        sankeyInfo = {}
        sankeyInfo['source'] = selectedFeatures[sourceNameInd]
        sankeyInfo['target'] = selectedFeatures[targetNameInd]
        sankeyInfo['value'] = counterValues[sourceNameInd][targetNameInd]  # np.corrcoef([signalData[:,sourceInd], signalData[:,targetInd]])[0][1]
        
        storeSankeyInfo.append(sankeyInfo)        
    
    Sankey.api_key("ff69182a-324b-46b9-90c5-77a6da9bd050")

    Sankey(
        storeSankeyInfo,
        thumbs_margin=1,
        popup_width=1000,
        arc_numbers=True,
        animated_intro=True,
        data_table_show_indices=False,
        title="Feature Pairs Explaining at Least 50% of the Stress Scores",
    ).to_html()

        
    
    
    if False:
        numFeaturesCombine = 1
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNames), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
        modelScores1, featureNames_Combinations1 = performMachineLearning.analyzeFeatureCombinations(signalData, signalLabels, featureNames, numFeaturesCombine, saveData = False, printUpdateAfterTrial = 15000, scaleY = testStressScores)
       
        featureNamesPermute_Good1 = featureNames_Combinations1[np.array(modelScores1) >= -10000]   
        
        featureCounts = np.zeros(len(selectedFeatures))
        for feature in featureNamesPermute_Good1:
            featureCounts[findType(feature.split('_')[-1])] += 1
            

     
     

     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     

     
     
     
     
     
     
     
     
     
     
     
     
     
  
   
   
   

