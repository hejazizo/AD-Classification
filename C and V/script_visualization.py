import matplotlib.pylab as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

## Loading dataset
alzheimer_df = pd.read_csv("data.csv")
y = alzheimer_df.output
X = alzheimer_df.drop(['output'], axis=1)

n_samples = X.shape[0]
n_features = X.shape[1]
n_classes = 2

print('dataset balance: {}'.format(sum(y)/y.shape[0]))
print('Number of Samples: {0}'.format(n_samples))
print('Number of features: {0}'.format(n_features))
print('Number of classes: {0}'.format(n_classes))

## Plotting Figures

#############################################
# path for output figures file
images_path = './images'
df = alzheimer_df
font = {'size': 10}
plt.rc('font', **font)

# specifies the parameters of our graphs
alpha_scatterplot = 0.2
alpha_bar_chart = 0.55
s = 100
markersize = 10

##################################
####     PLOTTING FIGURES     ####
##################################
fig = plt.figure()
### FIGURE 1: dataset balance
ax1 = plt.subplot2grid((1, 1),(0, 0))
# plots a bar graph of patients vs. controlled subjects
df.output.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax1.set_xlim(-1, 2)
plt.title("Samples Balance in AD Preprocessed Dataset (1 = Patient)")
plt.savefig('{}/dataset-balance.pdf'.format(images_path))

### FIGURE 2: age distribution
ax2 = plt.subplot2grid((1, 1),(0, 0))
plt.scatter(df.output, df.age, s=s, alpha=alpha_scatterplot)
plt.ylabel("Age")
plt.xticks((0, 1))
plt.xlim((-0.5, 1.5))
# formats the grid line style of our graphs
plt.grid(b=True, which='major', axis='y')
plt.title("AD by Age (1 = Patient)")
plt.savefig('{}/AD-distribution-age.pdf'.format(images_path))

### FIGURE 3: eTIV
ax3 = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df.output, df.eTIV/1000000, s=s, alpha=alpha_scatterplot)
plt.ylabel("eTIV (mm3)          [x1e6]")
plt.xticks((0, 1))
plt.xlim((-0.5, 1.5))
# formats the grid line style of our graphs
plt.grid(b=True, which='major', axis='y')
plt.title("AD by eTIV (1 = Patient)")
plt.savefig('{}/AD-eTIV.pdf'.format(images_path))

### FIGURE 4: Mask Volume
ax4 = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df.output, df.MaskVol/1000000, s=s, alpha=alpha_scatterplot)
# sets the y axis lable
plt.ylabel("Brain Mask Volume (mm3)          [x1e6]")
plt.xticks((0, 1))
plt.xlim((-0.5, 1.5))
# formats the grid line style of our graphs
plt.grid(b=True, which='major', axis='y')
plt.title("AD by Brain Mask Volume (1 = Patient)")
plt.savefig('{}/AD-MaskVol.pdf'.format(images_path))

### FIGURE 5: White matter - AGE
ax5 = plt.subplot2grid((1, 1), (0, 0))
plt.plot(df.age[df.output == 1], df.CerebralWhiteMatterVol[df.output == 1]/100000, 'rx', markersize=markersize)
plt.plot(df.age[df.output == 0], df.CerebralWhiteMatterVol[df.output == 0]/100000, 'b+', markersize=markersize)

plt.legend(['Patient', 'Contolled Subject'])
plt.ylabel("Cerebral White Matter Volume (mm3) [x1e5]")
plt.xlabel("Age")
# formats the grid line style of our graphs
plt.grid(b=True, which='major', axis='y')
plt.title("Age vs. Cerebral White Matter Volume")

plt.savefig('{}/AD-Age-WhiteMatter.pdf'.format(images_path))


### FIGURE 6: Cortex Volume - AGE
ax6 = plt.subplot2grid((1, 1), (0, 0))
plt.plot(df.age[df.output == 1], df.CortexVol[df.output == 1]/100000, 'rx', markersize=markersize, label='x')
plt.plot(df.age[df.output == 0], df.CortexVol[df.output == 0]/100000, 'b+', markersize=markersize, label='+')

plt.legend(['Patient', 'Contolled Subject'])
plt.ylabel("Cortex Volume (mm3) [x1e5]")
plt.xlabel("Age")

# formats the grid line style of our graphs
plt.grid(b=True, which='major', axis='y')
plt.title("Age vs. Cortex Volume")
plt.savefig('{}/AD-Age-CortexVol.pdf'.format(images_path))


### Figure 7: 3D image of 3 most important features
ax = plt.axes(projection='3d')

x = df['Right-Inf-Lat-Vent']
y = df['Right-Amygdala']
z = df['Left-Hippocampus']

ax.set_xticks((0, 2000, 4000, 6000))
ax.set_yticks((1000, 1500, 2000, 2500))

ax.scatter(x[df.output == 1], y[df.output == 1], z[df.output == 1], c='r', marker='^', linewidth=0.5);
ax.scatter(x[df.output == 0], y[df.output == 0], z[df.output == 0], c='b', marker='o', linewidth=0.5);

ax.set_xlabel('Right-Inf-Lat-Vent')
ax.set_ylabel('Right-Amygdala')
ax.set_zlabel('Left-Hippocampus')
ax.legend(['Patient', 'Control Subject'])

plt.savefig('{}/3d.pdf'.format(images_path))


