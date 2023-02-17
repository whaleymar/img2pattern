import numpy as np
import imageio
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
	# print(col.shape, palette.shape)
	# distances = np.sum((palette-col)**2, axis=1)
	distances = np.sum(((palette-col)*np.array([.3, .59, .11]))**2, axis=1)
	# if np.random.randint(0,100)==0: print(col,palette[np.argmin(distances)])
	return palette[np.argmin(distances)]

def getPNG(path: str) -> np.ndarray:
	# returns numpy array of image using imageio
	return imageio.imread(path)

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



# def main(RP_path, palette_file, dest_name):
# 	palette = readPalette(palette_file)
# 	dest_path = make_directory_skeleton(RP_path,dest_name)
# 	textures_path = os.path.join(RP_path, "assets/minecraft/textures")
# 	convert_textures(textures_path, dest_path, palette) # textures_path and dest_path start in same relative location


# if __name__=="__main__":
# 	# f = 'cobblestone.png'
# 	# p = "textures/block"
# 	# dp = "testing"
# 	# palette=readPalette("palette.txt")
# 	# palettize(f,p,dp,palette)
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("--RP_path",type=str,default="DefaultTP", help="path to texture pack you're palettizing (default: 'DefaultTP' (default MC texture pack))")
# 	parser.add_argument("--palette_file",type=str,default="palette.txt", help="palette file name")
# 	parser.add_argument("--rec_pack_name",type=str, default="New Texture Pack",help="name of your palettized resource pack (Default: 'New Texture Pack')")
# 	args = parser.parse_args()

# 	main(args.RP_path,args.palette_file,args.rec_pack_name)