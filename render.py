#TODO:
### custom symbols
### experiment with color-reducing algorithm
### reducing amt of greys, casting some grey spectrum to black
### prettify legend


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import util
import os
import numpy as np
from PIL import Image
import cv2

# struct for convenience
class grid():
	def __init__(self, fig, ax, gridsize, dpi):
		self.plts = (fig,ax)
		self.gsize = gridsize
		self.dpi = dpi

#lookup table for pattern marks
# markers = ['\\alpha', '\\beta', '\gamma', '\sigma','\infty', \
#             '\spadesuit', '\heartsuit', '\diamondsuit', '\clubsuit', \
#             '\\bigodot', '\\bigotimes', '\\bigoplus', '\imath', '\\bowtie', \
#             '\\bigtriangleup', '\\bigtriangledown', '\oslash' \
#            '\ast', '\\times', '\circ', '\\bullet', '\star', '+', \
#             '\Theta', '\Xi', '\Phi', \
#             '\$', '\#', '\%', '\S']
markers = ['2','6', '@','\\bigtriangleup','&','3','7','0','+','=','/','\\heartsuit',
			'\\star','\diamondsuit','\\Delta','\pi', '\Gamma','\lambda', '\odot', '\Cup', '\Join'
] # note: '2' is usally always black and '6' is always white 

dest_dir = 'output/'
save = True

def main():
	src_path = 'charizard_gen2.png'
	if not os.path.exists(dest_dir):
		os.mkdir(dest_dir)
	dest_path = os.path.join(dest_dir, os.path.basename(src_path)).split('.')[0] + '.pdf'

	dpi = 100
	maxsize=80
	maxcols = 20 # cat says 15-20
	path = os.path.join(os.getcwd(), src_path)
	# path = os.path.join(os.getcwd(), 'totodile4.png')
	threads = 'threadcolors.csv'
	
	img = util.getPNG(path)
	img = util.crop(img)
	gridsize = min(max(img.shape[0], img.shape[1]), maxsize) # setting max of 100 pixels for now

	if img.shape[0]>gridsize:
		img = cv2.resize(img, dsize=(gridsize, gridsize), interpolation=cv2.INTER_AREA) # this one looks so much better than any PIL algo

	imgpal = util.makePalette([img])
	pal, lookup, code2rgb = load_colors(threads)

	emptymask = img[:,:,3]==0 # store mask where alpha=0 before destroying 4th channel
	# show(img)
	img = img[:,:,:3]

	if imgpal.shape[0] > maxcols:
		img = reduce_colors(img, maxcols) # this is crazy slow. needs to be overhauled
		# show(img)
	
	img = util.palettize(img, pal)
	# show(img)
	
	pattern_c, pattern_m, mark2col = markup(img, pal, emptymask, maxcols, lookup)
	patches = legend(mark2col, code2rgb)

	g = init_grid(gridsize, dpi)
	pal = pal/255
	apply_syms(g, pattern_c, pattern_m, pal, onlycolor=False)

	# set legend location
	box = g.plts[1].get_position()
	g.plts[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

	g.plts[1].legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
	

	if not save:
		plt.show()
	else:
		plt.savefig(dest_path)
		print(f"saved figure to {dest_path}")

	return

def init_grid(gridsize:int, dpi:int) -> grid:
	# create matplotlib grid
	fig,ax = plt.subplots(1)

	markevery = 10
	inches = 10

	fig.set_size_inches(inches, inches, forward=True)
	fig.set_dpi(dpi)

	gridsize_og = gridsize
	gridsize = gridsize + (markevery - gridsize % markevery)
	plt.xlim(0,gridsize)
	plt.ylim(0,gridsize)
	ax.set_aspect('equal', adjustable='box')

	# set gridlines at integer intervals
	plt.xticks(ticks=list(range(gridsize)))
	plt.yticks(ticks=list(range(gridsize)))

	# config grid lines
	ax.grid(visible=True, markevery=1)
	xGridLines = ax.get_xgridlines()
	yGridLines = ax.get_ygridlines()
	for i in range(gridsize):
		xGridLines[i].set_color('black')
		yGridLines[i].set_color('black')
		if i % markevery != 0: 
			xGridLines[i].set_linewidth(0.3)
			yGridLines[i].set_linewidth(0.3)
		else:
			xGridLines[i].set_linewidth(1.5)
			yGridLines[i].set_linewidth(1.5)

	# remove labels
	labels = ['' if i % markevery != 0 else str(i) for i in range(gridsize)]
	ax.set_xticklabels(labels)
	ax.set_yticklabels(['']+labels[::-1])
	ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

	# remove ticks
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')

	g = grid(fig, ax, gridsize_og, dpi)
	return g

# converts each pixel in `img` into a palette index
# assumes every pixel in `img` is in palette
def markup(img:np.ndarray, pal:np.ndarray, emptymask:np.ndarray, 
		maxcols:int, lookup:dict) -> tuple:
	coltracker={} # maps palette index to mark index
	num = 2


	# get brightest and darkest colors in image
	imgpal = np.unique(img[~emptymask].reshape(-1,3), axis=0)
	palsum = imgpal.sum(axis=1)
	maxcol = imgpal[np.argmax(palsum)]
	mincol = imgpal[np.argmin(palsum)]

	palette_ix = list()
	sym_ix = list()
	for i in range(img.shape[0]):
		pnewrow = list()
		snewrow = list()
		for j in range(img.shape[1]):
			if emptymask[i,j]: 
				pnewrow.append(0)
				snewrow.append(0)
			else:
				col = img[i,j,:]
				# +1 offset so we don't share index with alpha=0
				col_ix = (pal==col[:3]).all(axis=1).nonzero()[0].item() + 1
				
				if col_ix not in coltracker.keys():
					if (col==mincol).all():
						coltracker[col_ix] = 0
					elif (col==maxcol).all():
						coltracker[col_ix] = 1
					else:
						coltracker[col_ix] = num
						num+=1
				snewrow.append(coltracker[col_ix])
			
				pnewrow.append(col_ix) 
				
		palette_ix.append(pnewrow)
		sym_ix.append(snewrow)

	reversedmap = {v:k for k,v in coltracker.items()} # maps mark index to palette index
	mark2col = [lookup[reversedmap[i]-1] for i in range(len(coltracker))]
	return np.array(palette_ix), np.array(sym_ix), mark2col

def apply_syms(g:grid, pattern_c:np.ndarray, pattern_m:np.ndarray, palette:np.ndarray, onlycolor:bool = False) -> None:
	for i in range(g.gsize):
		for j in range(g.gsize):
			ix = pattern_c[i,j]
			ix2 = pattern_m[i,j]
			if ix==0: continue
			add_sym(j,i,g,markers[ix2],palette[ix-1],onlycolor)

def add_sym(x:int, y:int, g:grid, mk:str, col:np.ndarray, onlycolor:bool) -> None:
	# plot a single symbol on the grid, and fill the cell with its color
	fig,ax = g.plts
	mksize = 3.5
	offsetx1 = 0.45
	offsetx2 = 0.45 + (mksize/g.dpi)
	offsety1 = 0.5
	offsety2 = 0.5 - (mksize/g.dpi)
	alpha_col = 0.4 if not onlycolor else 1.0
	
	# convert from numpy index to plot point
	y = g.gsize - y - 1

	# add symbol:
	if not onlycolor:
		# markcol = (38/255, 39/255, 41/255)
		markcol = (0.0, 0.0, 0.0)
		ax.plot(x+offsetx1, y+offsety1, marker = f'${mk}$', markersize=mksize,
			markeredgewidth=0.2,
			markeredgecolor=markcol, markerfacecolor=markcol, alpha = 1.0)

	# color cell:
	mksize = g.dpi/g.gsize
	ax.plot(x+offsetx2, y+offsety2, marker = 's', markersize=mksize*3.3,
		markeredgewidth=0, markerfacecolor=tuple(col), alpha = alpha_col)

def reduce_colors(img:np.ndarray, maxcols:int) -> np.ndarray:
	# reduces img to use only maxcols colors
	
	# mc_pal = Image.fromarray(img).convert('P', palette=Image.ADAPTIVE, colors=maxcols) # to palette
	# img = np.array(
	# 	mc_pal.convert('RGB', palette=Image.ADAPTIVE, colors=maxcols) # to RGB
	# 	)

	# trying Image.quantize so I get more palette options
	im = Image.fromarray(img)
	img = im.quantize(colors=maxcols, 
		method=1, dither=3, palette=im.palette
		)
	img = np.array(
		img.convert('RGB', palette=im.palette)
		)
	return img

def load_colors(filename:str) -> tuple:
	# reads thread color file and returns palette as numpy array
	table = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None)[1:,:]
	pal = table[:,2:5].astype(int)
	lookup = {i: (table[i,0],table[i,1]) for i in range(table.shape[0])}
	code2rgb = {table[i,0]: tuple(table[i,2:5].astype(float)/255) for i in range(table.shape[0])}
	return pal,lookup,code2rgb

def show(img:np.ndarray) -> None:
	plt.imshow(img)
	plt.show()
	return

def legend(mark2col:dict, code2rgb) -> None:
	patches = []
	for i in range(len(mark2col)):
		code, mark = mark2col[i]
		print(markers[i], code, mark)
		patches.append(mpatches.Patch(color=code2rgb[code], label = f"${markers[i]}$\t({code})"))
	
	return patches



def to_clipboard(arr):
	import pandas 
	df = pandas.DataFrame(arr)
	df.to_clipboard(index=False,header=False)

if __name__ == '__main__':
	main()