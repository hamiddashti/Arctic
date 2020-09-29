import numpy as np
from sklearn.metrics import confusion_matrix
import rasterio
import fiona





year1 = np.array([
	[1,1,2,3],
	[1,1,4,3],
	[2,1,1,3],
	[1,2,1,3]])
year2 = np.array([
	[2,2,1,3],
	[4,1,2,3],
	[1,3,1,3],
	[1,4,1,3]])

confusion_matrix(true, pred)
