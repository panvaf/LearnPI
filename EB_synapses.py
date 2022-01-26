"""
Analyze synapses to E-PG neurons in the EB from the fly connectome.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from pathlib import Path
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np

# Fontsize appropriate for plots
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)      # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)   # fontsize of the figure title

# Body IDs of all available neurons
EPG_bodyIDs = [416642425, 912545106, 1127476096, 819828986, 912601268, 1035045015,
               1447576662, 387364605, 665314820, 5813014873, 695629525, 478375456,
               725951521, 910438331, 1034219901, 1126647624]

# Directories
parent_path = str(Path(os.getcwd()).parent)
data_path = parent_path + '\\Connectome Synapses\\'
load_path = parent_path + '\\savefiles\\'

try:
    # Load crossvalidation results
    scores = np.load(load_path + 'SVM3Drbf.npz')['scores']
    svm = np.nan
except:
    # Support Vector Classifier
    svm = SVC(kernel="rbf")
    
    # Number of nested crossvalidation svm runs per neuron
    n_trial = 30
    
    # Set up possible values of parameters to optimize over
    p_grid = {"C": [.3, 1, 3],
              "gamma": [.3, 1, 3]}
    
    scores = np.zeros((len(EPG_bodyIDs),n_trial))

# 2-D plot with synapse locations for all neurons tested

fig, axs = plt.subplots(4,4,sharex=False,sharey=False,figsize = (6.5,5))

for num, EPG_bodyID in enumerate(EPG_bodyIDs):
    
    # Load data, removing duplicate synapses
    R2_to_EPG = pd.read_csv(data_path + 'ER2 to E-PG (' + str(EPG_bodyID) + ').csv').drop_duplicates()
    R4d_to_EPG = pd.read_csv(data_path + 'ER4d to E-PG (' + str(EPG_bodyID) + ').csv').drop_duplicates()
    PEN1_to_EPG = pd.read_csv(data_path + 'PEN1 to E-PG (' + str(EPG_bodyID) + ').csv').drop_duplicates()
    PEN2_to_EPG = pd.read_csv(data_path + 'PEN2 to E-PG (' + str(EPG_bodyID) + ').csv').drop_duplicates()
    
    if not np.isnan(svm):
        # Create data for SVMs
        proximal_syn = R2_to_EPG.append(R4d_to_EPG, ignore_index=True); proximal_syn['class'] = 0
        distal_syn = PEN1_to_EPG.append(PEN2_to_EPG, ignore_index=True); distal_syn['class'] = 1
        all_syn = proximal_syn.append(distal_syn, ignore_index=True)
        X_syn = StandardScaler().fit_transform(all_syn.iloc[:,0:3])
        y_syn = all_syn.iloc[:,3]
        
        for k in range(n_trial):
            
            # nested crossvalidation
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=k)
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=k)
            
            clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
            score = cross_val_score(clf, X=X_syn, y=y_syn, cv=outer_cv)
            scores[num,k] = score.mean()
        
    i = num // 4
    j = num % 4
    
    # Plot
    axs[i,j].scatter(PEN1_to_EPG[["syn.location.y"]], PEN1_to_EPG[["syn.location.z"]], c='green', marker='o',s=.2)
    axs[i,j].scatter(PEN2_to_EPG[["syn.location.y"]], PEN2_to_EPG[["syn.location.z"]], c='darkorange', marker='o',s=.2)
    axs[i,j].scatter(R2_to_EPG[["syn.location.y"]], R2_to_EPG[["syn.location.z"]], c='mediumorchid', marker='o',s=.2)
    axs[i,j].scatter(R4d_to_EPG[["syn.location.y"]], R4d_to_EPG[["syn.location.z"]], c='dodgerblue', marker='o',s=.2)
    axs[i,j].set_xticks([])
    axs[i,j].set_yticks([])
    axs[i,j].spines['top'].set_visible(False)
    axs[i,j].spines['right'].set_visible(False)
    if i==3 and j==0:
        axs[i,j].set_xlabel('Y dimension')
        axs[i,j].set_ylabel('Z dimension')
        axs[i,j].set_title('Neuron ID = {}'.format(EPG_bodyID))
    else:
        axs[i,j].set_title('{}'.format(EPG_bodyID))
    
# Legend
plt.scatter([],[], c='green', marker='o', label = '$W^{HR}$ (P-EN1)',s=.2)
plt.scatter([],[], c='darkorange', marker='o', label = '$W^{rec}$ (P-EN2)',s=.2)
plt.scatter([],[], c='mediumorchid', marker='o', label = '$I^{vis}$ (R2)',s=.2)
plt.scatter([],[], c='dodgerblue', marker='o', label = '$I^{vis}$ (R4d)',s=.2)  
fig.legend(frameon=False,loc='upper center', bbox_to_anchor=(0.5, 1.05),markerscale=10,ncol=4)
plt.tight_layout()

# 2-D plot with SVM nested crossvalidation results per neuron

fig, axs = plt.subplots(4,4,sharex=True,sharey=True,figsize = (3.7,2.5))

for num, EPG_bodyID in enumerate(EPG_bodyIDs):
    
    i = num // 4
    j = num % 4
    
    # Plot
    axs[i,j].hist(scores[num,:],bins=np.linspace(np.min(scores),np.max(scores),25))
    axs[i,j].spines['top'].set_visible(False)
    axs[i,j].spines['right'].set_visible(False)
    axs[i,j].spines['left'].set_position(('data', 0.95))
    if i==3 and j==0:
        axs[i,j].set_xlabel('Test accuracy')
        axs[i,j].set_ylabel('Count')
        axs[i,j].set_xticks([0.96,0.99])
        axs[i,j].xaxis.set_minor_locator(MultipleLocator(0.975))
        axs[i,j].set_yticks([0,10])
        axs[i,j].yaxis.set_minor_locator(MultipleLocator(5))

plt.tight_layout()