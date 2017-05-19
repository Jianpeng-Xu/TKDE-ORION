Description of data:
1. The data used in this work is under folder ./data
2. There are 12 files in the data folder, each of which contains the varilables and information for one river basin in US.
3. Variables include: X, y, time, ensembledMean.

Description of code:
1. ORION-epsilon:
   ORION.m --- ORION update for each round
   restartUpdate.m --- This code include the restart strategy to deal with the missing observations in target window. In order to simulate the real world scenario, the missing value is generated in the code. 
   mainORION.m --- This is the main code for running the process. This code can be modified to do parameter selection. 
2. ORION-QR
   Similar code structures to ORION-epsilon. The ORION-QR.m is written using CVX package. Please install CVX before running ORION-QR.