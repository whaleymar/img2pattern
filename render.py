#TODO:
### use thread colors as input palette
### print required colors in output


import matplotlib.pyplot as plt
import util
import os
import numpy as np
import sys
from PIL import Image
import cv2

# struct for convenience
class grid():
	def __init__(self, fig, ax, gridsize, dpi):
		self.plts = (fig,ax)
		self.gsize = gridsize
		self.dpi = dpi

#lookup table for pattern marks
markers = ['\\alpha', '\\beta', '\gamma', '\sigma','\infty', \
            '\spadesuit', '\heartsuit', '\diamondsuit', '\clubsuit', \
            '\\bigodot', '\\bigotimes', '\\bigoplus', '\imath', '\\bowtie', \
            '\\bigtriangleup', '\\bigtriangledown', '\oslash' \
           '\ast', '\\times', '\circ', '\\bullet', '\star', '+', \
            '\Theta', '\Xi', '\Phi', \
            '\$', '\#', '\%', '\S']


def main():
	dpi = 100
	maxsize=60
	# path = os.path.join(os.getcwd(), 'totodile4.png')
	path = os.path.join(os.getcwd(), 'hatenna.png')
	
	img = util.getPNG(path)
	gridsize = min(max(img.shape[0], img.shape[1]), maxsize) # setting max of 100 pixels for now

	if img.shape[0]>gridsize:
		# pimg = Image.fromarray(img).resize((gridsize,gridsize), resample=Image.LANCZOS)
		# img = np.array(pimg)
		img = cv2.resize(img, dsize=(gridsize, gridsize), interpolation=cv2.INTER_AREA)

	imgpal = util.makePalette([img])
	pal = imgpal # can replace this with input palette later

	# store mask where alpha=0 before destroying channel:
	emptymask = img[:,:,3]==0

	if imgpal.shape[0] > len(markers):
		# plt.imshow(img)
		# plt.show()
		img = img[:,:,:3]
		# print('image shape: ',img.shape)
		# print(img[40:45, 40:45, :])
		# plt.imshow(img)
		# plt.show()
		img = reduce_colors(img, len(markers))
		# plt.imshow(img)
		# plt.show()
		img = util.palettize(img, pal)
		to_clipboard(np.unique(img.reshape(-1,3),axis=0))
		# plt.imshow(img)
		# plt.show()
		# print('image shape: ',img.shape)
		# print(img[40:45, 40:45, :])
		# print(imgpal.shape, util.makePalette([img]).shape)
	
	pattern_c, pattern_m = markup(img, pal, emptymask, len(markers))
	# print(pattern_c.shape, pattern_m.shape)
	
	g = init_grid(gridsize, dpi)
	pal = pal/255
	apply_syms(g, pattern_c, pattern_m, pal)
	plt.show()

	return

# TODO remove ticks
def init_grid(gridsize:int, dpi:int) -> grid:
	# create matplotlib grid
	fig,ax = plt.subplots(1)

	ax.grid(visible=True, markevery=1)

	fig.set_size_inches(12, 12, forward=True)
	fig.set_dpi(dpi)

	plt.xlim(0,gridsize)
	plt.ylim(0,gridsize)
	ax.set_aspect('equal', adjustable='box')

	plt.xticks(ticks=list(range(gridsize)))
	plt.yticks(ticks=list(range(gridsize)))

	ax.set_xticklabels([])
	ax.set_yticklabels([])

	g = grid(fig, ax, gridsize, dpi)
	return g

# converts each pixel in `img` into a palette index
# assumes every pixel in `img` is in palette
def markup(img:np.ndarray, pal:np.ndarray, emptymask:np.ndarray, maxcols:int = -1) -> np.ndarray:
	if maxcols>0:
		coltracker={}
		num = 0
	else:
		coltracker=None
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
				if maxcols>0:
					if col_ix not in coltracker.keys():
						coltracker[col_ix] = num
						num+=1
					snewrow.append(coltracker[col_ix])
				else:
					snewrow.append(col_ix) 
			
				pnewrow.append(col_ix) 
				
		palette_ix.append(pnewrow)
		sym_ix.append(snewrow)
		# print(len(pnewrow), len(snewrow))
	return np.array(palette_ix), np.array(sym_ix)

def apply_syms(g:grid, pattern_c:np.ndarray, pattern_m:np.ndarray, palette:np.ndarray) -> None:
	for i in range(g.gsize):
		for j in range(g.gsize):
			ix = pattern_c[i,j]
			ix2 = pattern_m[i,j]
			if ix==0: continue
			add_sym(j,i,g,markers[ix2],palette[ix-1])

def add_sym(x:int, y:int, g:grid, mk:str, col:np.ndarray) -> None:
	# plot a single symbol on the grid, and fill the cell with its color
	fig,ax = g.plts
	mksize = 4
	offset = 0.5 + (mksize/g.dpi)
	
	# convert from numpy index to plot point
	y = g.gsize - y - 1

	# add symbol:
	markcol = (38/255, 39/255, 41/255)
	ax.plot(x+offset, y+offset, marker = f'${mk}$', markersize=mksize,
		markeredgewidth=0.2,
		markeredgecolor=markcol, markerfacecolor=markcol, alpha = 1)

	# color cell:
	mksize = g.dpi/g.gsize
	ax.plot(x+offset, y-offset, marker = 's', markersize=mksize*6.5,
		markeredgewidth=0, markerfacecolor=tuple(col), alpha = 0.5)

def reduce_colors(img:np.ndarray, maxcols:int) -> np.ndarray:
	# reduces img to use only maxcols colors
	# print(maxcols)
	mc_pal = Image.fromarray(img).convert('P', palette=Image.ADAPTIVE, colors=maxcols)
	# print(np.min(mc_pal), np.max(mc_pal))
	img = mc_pal.convert('RGB', palette=Image.ADAPTIVE, colors=maxcols)
	img=np.array(img)
	# mc_pal=np.array(mc_pal)
	# print(mc_pal[40:45,40:45], img[40:45,40:45,:], sep='\n')
	return img

def to_clipboard(arr):
	import pandas 
	df = pandas.DataFrame(arr)
	df.to_clipboard(index=False,header=False)

if __name__ == '__main__':
	main()