from pylab import *
import matplotlib as plt
import numpy as np

from fig_style import *

def B(N, k):
    return np.cos(np.pi * k / (N+1))

nrows = 2
ncols = 2

mew = 0.5
msz = 4

gold_mean = (np.sqrt(5.0)-1.0)/2.0
aspect_ratio = gold_mean*1.2
scale = 1
fig_width_pt = 246.0
inches_per_pt = 1.0/72.27
fig_width = fig_width_pt*inches_per_pt*scale
fig_height = aspect_ratio*fig_width

fig = figure(figsize=(fig_width*float(ncols), fig_height*float(nrows)))
G = GridSpec(nrows, ncols)

ax = {}
al = {}
for i in range(0, nrows):
	ax[i] = {}
	for j in range(0, ncols):
		ax[i][j] = subplot(G[i:(i+1), j:(j+1)])

colorsA = cm.Set1(np.linspace(0,1,9))[1:]
colorsB = cm.tab10(np.linspace(0,1,10))[5:]
colorsC = cm.tab10(np.linspace(0,1,10))
symbols = ['o', 's', 'v', '^', '>', 'D', '*']


#############################
# A
#############################
ca = ax[0][0]
Ns = [500, 502, 504, 506]
for i, N in enumerate(Ns):
	data = np.loadtxt(f'data/clustering_vs_i/N_{N}.dat')
	ca.plot(data[:,0]-8, data[:,1], c=colorsA[i], marker=symbols[i], markeredgewidth=mew, markeredgecolor='k', ms=msz if i != 0 else msz+1, label=f'$N = {N}$')
Ns = [503, 505, 507]
for i, N in enumerate(Ns):
	data = np.loadtxt(f'data/clustering_vs_i/N_{N}.dat')
	ca.plot(data[:,0]-8, data[:,1], c=colorsA[i], marker=None, ls='--', lw=0.6)

lines, labels = ca.get_legend_handles_labels()
ca.legend(lines, labels, loc='upper left', shadow=False, fancybox=False, frameon=True, numpoints=1, ncol = 2, columnspacing = 1, handlelength = 1)


ca.set_xlabel(r'Shifted spin index $i - i_c$')
ca.set_ylabel(r'$c_{i - i_c}$')
#ca.set_xscale("log")
#ca.set_yscale("log")
#ca.set_xbound(0.04, 0.6)
ca.set_ybound(0.01, 0.16)


#############################
# C - E
#############################
files = ['S_3.5_P_7.dat', 'S_3.8_P_19.dat', 'S_3.1_P_31.dat']
ax_idx = [(0, 1), (1, 0), (1, 1)]
for i, f in enumerate(files):
	ca = ax[ax_idx[i][0]][ax_idx[i][1]]
	data = np.loadtxt('data/mean_clustering_vs_N/'+f)
	ca.plot(data[:,0], data[:,1], c=colorsA[i], marker=symbols[i], markeredgewidth=mew, markeredgecolor='k', ms=msz-2, lw=0.6)
	yl = ca.get_ylim()
	p = int(f.split('_')[3].split('.')[0])
	for i in range(0, 5):
		xc = data[0,0]+i*p
		ca.plot([xc, xc], yl, c=colorsA[3], ls='--')

	ca.set_xlabel(r'$N$')
	ca.set_ylabel(r'$\langle c_{i - i_c} \rangle$')
	#ca.set_xscale("log")
	#ca.set_yscale("log")
	#ca.set_xbound(0.04, 0.6)
	#ca.set_ybound(0.01, 0.16)



#############################
# Letters on corners
#############################
# for i in range(nrows):
# 	for j in range(ncols):
# 		#ax[i][j].text(-0., 1.01,r'{\font\elbold=phvb at 9pt \elbold '+chr(ord('`') + (ncols * i + j + 1))+'}', horizontalalignment='left',verticalalignment='bottom', transform=ax[i][j].transAxes)
# 		ax[i][j].text(-0., 1.02, r'('+chr(ord('`') + (ncols * i + j + 1))+')', horizontalalignment='left',verticalalignment='bottom', transform=ax[i][j].transAxes)

format_axes([ax[i][j] for i in range(nrows) for j in range(ncols)])

# Save figure
tight_layout()
#G.tight_layout(fig, rect=[0, 0, 1, 0.96], pad = 0.3)
fig.savefig('figures/fig_size-periodicity_2x2.pdf', bbox_inches='tight')


