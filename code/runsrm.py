import argparse
import numpy as np
import os

"""
Assumes terminal command "python" will run python2 on your system. (Or whatever python runs the srm for you.)
Assumes that num_regions x num_timepoints is shared across input data.
Regular SRM command: python run_algo_only.py name dataset srm iterations k
"""

# Example call of this file:
# python runsrm.py ./run_algo_only.py srmsimple ./input/ ./output/ ./data/notschizo.npy ./data/schizo.npy ./data/singletestpatient.npy ex1 5 5 10 10

parser = argparse.ArgumentParser()
parser.add_argument("srmfile") # path to srm algorithm python file
parser.add_argument("algo") # algorithm to use
parser.add_argument("inputpath") # path to directory srm algorithm looks for input files
parser.add_argument("outputpath") # path to directory srm algorithm stores results
parser.add_argument("healthy") # path to input data of healthy patients (this file converts from Roberto's form to the form they want)
parser.add_argument("schizophrenia") # path to input data of schizophrenic patients (this file converts from Roberto's form to the form they want)
parser.add_argument("querypatients") # path to input data of patients we want to classify
parser.add_argument("name") # name for this test
parser.add_argument("niter_1") # number of iterations for the algorithm to run for the first run
parser.add_argument("niter_2") # number of iterations for the algorithm to run for the first run
parser.add_argument("nfeature_1") # length of shared response vectors for the first run (k_1)
parser.add_argument("nfeature_2") # length of shared response vectors for the second run (k_2)

args = parser.parse_args()

# Extract the initial data and get some patients counts
initialHealthyPatients = np.load(args.healthy)
initialSchizophrenicPatients = np.load(args.schizophrenia)
initialQueryPatients = np.load(args.querypatients)

healthyCount = len(initialHealthyPatients)
schizophrenicCount = len(initialSchizophrenicPatients)
queryPatientsCount = len(initialQueryPatients)

print("Healthy Count: " + str(healthyCount))
print("Schizophrenic Count: " + str(schizophrenicCount))
print("Query Patient Count: " + str(queryPatientsCount))

# Load data into the array for the first test
initialDataExtracted = []

for i in range(healthyCount):
    initialDataExtracted.append(initialHealthyPatients[i])

for i in range(schizophrenicCount):
    initialDataExtracted.append(initialSchizophrenicPatients[i])

for i in range(queryPatientsCount):
    initialDataExtracted.append(initialQueryPatients[i])

print("Total: " + str(len(initialDataExtracted)))

# Save it into the directory that srm looks in
np.savez(args.inputpath+args.name+"_initial.npz", data=initialDataExtracted)

# Run first experiment
# TODO support other flags
os.system("python " + args.srmfile + " " + args.name + "_initial " + args.name + "_initial.npz " + " " + args.algo + " " + str(args.niter_1) + " " + str(args.nfeature_1))
print("Done SRM on full dataset")

# Load output file
# TODO support other types of alignment algorithm
initialDataOutput = np.load(args.outputpath + args.name + "_initial/" + args.algo + "/" + str(args.nfeature_1) + "feat/identity/" + args.algo + str(args.niter_1) + "_WS.npz")

# Form the residual
sharedResponseComputedData = initialDataOutput['W'].dot(initialDataOutput['S'])

# Load data for the healthy residual
# TODO support different num_regions
healthyResidual = []
for i in range(healthyCount):
    temp = []
    for j in range(len(initialHealthyPatients[i])):
        temp.append(np.subtract(initialHealthyPatients[i][j], sharedResponseComputedData[(i * len(initialHealthyPatients[i])) + j]))
    healthyResidual.append(temp)

# Save it into the directory that srm looks in
np.savez(args.inputpath+args.name+"_healthy.npz", data=healthyResidual)

# Run second experiment
# TODO support other flags
os.system("python " + args.srmfile + " " + args.name + "_healthy " + args.name + "_healthy.npz " + " " + args.algo + " " + str(args.niter_2) + " " + str(args.nfeature_2))
print("Done SRM on healthy residuals")

# Load data for the schizophrenic residual
# TODO support different num_regions
schizophrenicResidual = []
for i in range(schizophrenicCount):
    temp = []
    for j in range(len(initialSchizophrenicPatients[i])):
        temp.append(np.subtract(initialSchizophrenicPatients[i][j], sharedResponseComputedData[((i + healthyCount) * len(initialSchizophrenicPatients[i])) + j]))
    schizophrenicResidual.append(temp)

# Save it into the directory that srm looks in
np.savez(args.inputpath+args.name+"_schizophrenic.npz", data=schizophrenicResidual)

# Run third experiment
# TODO support other flags
os.system("python " + args.srmfile + " " + args.name + "_schizophrenic " + args.name + "_schizophrenic.npz " + " " + args.algo + " " + str(args.niter_2) + " " + str(args.nfeature_2))
print("Done SRM on schizophrenic residuals")
