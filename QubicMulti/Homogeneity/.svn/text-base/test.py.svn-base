# Some initial definitions
# If you want to run this at home
# you need to change these file locations
polyfile = '/Users/hamilton/SDSS/Data/LRG/current_boss_geometry.ply'
targetfile = '/Users/hamilton/SDSS/Data/LRG/FinalFits/final-boss9.fits'
spallfile = '/Users/hamilton/SDSS/Data/LRG/spAll-v5_4_31.fits'
steradian = (np.pi/180)**2



# first, some polygon manipulation
import mangle
import mangle_utils
import Match
import pairs.mr_wpairs as wpairs
# Initialize a Mangle instance with a polygon file for one chunk.
mng = mangle.Mangle(polyfile)

# Plot all the polygons
polys = mng.graphics()
for poly in polys:
    plot(poly['ra'],poly['dec'],lw=2)

# all polygons with areas greater than 1 sq. deg.
test = mng.areas > 3. * (np.pi/180)**2
polys = mng[test].graphics()
for poly in polys:
    plt.plot(poly['ra'],poly['dec'],lw=2)







