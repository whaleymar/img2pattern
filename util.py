import numpy as np
from PIL import Image
import os

def getFiles(path: str) -> list:
	# returns list of all png files in path

	return [f for f in os.listdir(path) if f[-4:]=='.png']

def makePalette(imgs: list, palette_fname: str = "") -> np.ndarray:
	# iterates through list of files and compiles palette 
	# uses all PNG files in path
	# if palette_fname is given, will save palette to txt file with that name
	
	palette = None
	for arr in imgs:
		# print(arr.shape,arr.shape[2])
		if arr.shape[2]==4:
			arr = sanitize(arr.reshape(-1,4))
		else:
			arr = sanitize(arr.reshape(-1,3))
		uniques = np.unique(arr,axis=0)
		if palette is not None:
			palette = np.unique(np.concatenate([palette, uniques]),axis=0)
		else:
			palette = uniques

	if palette_fname:
		np.savetxt(palette_fname,palette,fmt='%i',delimiter=',')
	return palette

def sanitize(palette: np.ndarray) -> np.ndarray:
	# takes palette, removes any color with alpha=0 and removes 4th channel
	return palette[palette[:,-1]!=0][:,:3]

def findCol(col:np.ndarray, palette:np.ndarray) -> np.ndarray: # col should be np.array in form [R,G,B], palette should be array of cols
	# finds the color in "palette" that's closest to "col"
	distances = np.sum(((palette-col)*np.array([.299, .587, .114]))**2, axis=1)
	return palette[np.argmin(distances)]

def getPNG(path: str) -> np.ndarray:
	# returns numpy array of image
	result = np.array(Image.open(path).convert("RGBA"))
	return result

def readPalette(filepath: str) -> np.ndarray:
	# reads palette from disk
	if os.path.exists(filepath):
		palette = np.loadtxt(filepath,dtype=int,delimiter=",")
		if palette.shape[-1]==4:
			palette = palette[:,:3] # reshape if there's an alpha val
		return palette
	else:
		raise ValueError("palette file '{}' does not exist".format(filepath))

def palettize(img: np.ndarray, palette: np.ndarray) -> np.ndarray:
	recolored = np.apply_along_axis(findCol,2,img,palette) # TODO: parallelize this
	recolored = recolored.astype(np.uint8)

	return recolored

def crop(img:np.ndarray) -> np.ndarray:
	# crops image to remove empty space

	mask = ~img[:,:,3]==0
	filledrows = np.argwhere(np.sum(mask, axis=1)).reshape(-1) # row indexes with non-transparent pixels
	filledcols = np.argwhere(np.sum(mask, axis=0)).reshape(-1) # col indexes with  "       "        "
	l = np.min(filledrows)
	r = np.max(filledrows)+1
	t = np.min(filledcols)
	b = np.max(filledcols)+1

	w = r-l
	h = b-t
	bufy = 0 if w>=h else h-w
	bufx = 0 if h>=w else w-h
	return img[l:r+bufy, 
			   t:b+bufx, :]