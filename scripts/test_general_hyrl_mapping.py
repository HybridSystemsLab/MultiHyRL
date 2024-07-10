import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib
import pickle

from scipy.spatial import Voronoi, voronoi_plot_2d
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def cart2pol(z):
    x, y = z[0], z[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(z):
    rho, phi = z[0], z[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def transform_coordinates(position_to_transform, targeted_obstacle, reference_obstacle=(-2,0)):
    polar_position_to_transform = cart2pol(position_to_transform)
    polar_target_obstacle = cart2pol(targeted_obstacle)
    polar_reference_obstacle = cart2pol(reference_obstacle)
    scale_to_transform = polar_target_obstacle[0] / polar_reference_obstacle[0]
    angle_to_transform = polar_target_obstacle[1] - polar_reference_obstacle[1]
    transformed_polar = (polar_position_to_transform[0]/scale_to_transform, polar_position_to_transform[1]-angle_to_transform)
    transformed_cartesian = pol2cart(transformed_polar)
    return transformed_cartesian

def plot_Mi(Mij, index_lambda,
            fign=1, res=40, POSITIONAL_SET_POINT=(0,0), bound=7,
            plot_text=False):
    # j is either 0 or 1

    vor_plotting = Voronoi(Mij.vor.points)
    xp = np.linspace(-bound, bound, res)
    yp = np.linspace(-bound, bound, res)
    xi,yi = np.meshgrid(xp,yp)
    in_Mi0 = np.zeros((res,res))
    in_Mi1 = np.zeros((res,res))
    for idx in range(res):
        for idy in range(res):
            point = (xp[idx], yp[idy])
            in_Mi0[idy, idx] = Mij.get_in_Mi(point, index_q=0, index_lambda=index_lambda) 
            in_Mi1[idy, idx] = Mij.get_in_Mi(point, index_q=1, index_lambda=index_lambda)
    in_Mi01 = in_Mi0*in_Mi1
    voronoi_plot_2d(vor_plotting, show_vertices=False, line_colors='black', show_points=False,
                    line_width=1, line_alpha=1)
    plt.figure(fign)
    cmap = clrs.ListedColormap(['none', 'lightblue'])
    plt.scatter(xi, yi, s=15, c=in_Mi0, rasterized=True, cmap =cmap)
    cmap = clrs.ListedColormap(['none', 'lightgreen'])
    plt.scatter(xi, yi, s=15, c=in_Mi1, rasterized=True, cmap =cmap)
    cmap = clrs.ListedColormap(['none', 'lightcoral'])
    plt.scatter(xi, yi, s=15, c=in_Mi01, rasterized=True, cmap =cmap)
    for index, obstacle in enumerate(Mij.vor.points):
        x_obst, y_obst = obstacle[0], obstacle[1]
        obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                        radius=0.5, color='gray')
        if index < 2: # for two obstacle case plotting
            plt.text(x_obst,y_obst-0.1, str(index+1), fontsize=22, color='black', weight='bold',
            horizontalalignment='center', verticalalignment='center')
            plt.gca().add_patch(obstaclePatch)

    
    plt.plot(POSITIONAL_SET_POINT[0], POSITIONAL_SET_POINT[1], '*', color='red', zorder=15, markersize=12)
    plt.text(POSITIONAL_SET_POINT[0]+.1, POSITIONAL_SET_POINT[1]+.4, r'$p^*$', fontsize=22, color='red', zorder=15)
    if plot_text:
        plt.plot(-3, 5, 'o', color='black', markersize=6)
        plt.text(-2.9, 5.4, r'$p_1$', fontsize=22, color='black')
        plt.plot(1.5, -1.5, 'o', color='black', markersize=6)
        plt.text(1.6, -1.1, r'$p_2$', fontsize=22, color='black')
    plt.grid(visible=True)
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    # plt.xlim([-bound, bound])
    # plt.ylim([-bound, bound])
    plt.xlim([-5, 1])
    plt.ylim([-3, 3])
    plt.tight_layout()



cc = 1
res = 100
plot_fits = True
ex_number = 10

if ex_number == 10:
    NUMBER_OF_STATIC_OBSTACLES = 1
    L_ob = np.sqrt(1.5**2+1.5**2)
    STATIC_OBSTACLE_RADII = 0.5

load_name = 'hyrl_sets/ob1_Mij_x00_y00_static_example11_v1.pkl' 

with open(load_name, 'rb') as inp:
    Mij_general = (pickle.load(inp))   

obstacle_setting = [(2, 2), (-2, -2), (2, -2), (-2, 2)]
# obstacle_setting = [(-1.7, 2.6), (-2.4, -1.9), (-4, 1.4)]
# obstacle_setting = [(-1.7, 2.6),
#                                     (-2.4, -1.9),
#                                     (-4, 1.4)]
# obstacle_setting = [(2, 2), (-2, -2), (2, -2), (-2, 2), (2.5,0), (-2.5, 0), (0, 2.5), (0,-2.5)]
# obstacle_setting = [(1.0659098652892163, 1.76211725699931),
#  (-1.8418554867405013, 1.1279573887169625),
#  (-1.0578568182206252, -1.6664832419982727),
#  (1.7393365533607767, -0.8153839157303949),
#  (4.928382526031488, 0.8432352442275497),
#  (2.9800716318454814, -4.014868997746977),
#  (-1.521342452289482, -4.762931570247659),
#  (-3.9783682690481204, -3.028627728168496),
#  (-4.9774352586849595, -0.47448735030534683),
#  (-3.8570063202900293, 3.18174515718069),
#  (0.6260947951187125, 4.960645654299977),
#  (3.443030189194816, 3.625678297407688)]
# obstacle_setting = [(0.35057379297857, 2.188375927859161),
#  (-1.7134133806946483, 1.4575856669467404),
#  (-1.1589092842677644, -1.5661304581425155),
#  (1.7864247018307355, -1.0748871045596133),
#  (4.8139447220373865, 1.3512720722224645),
#  (0.9778348766807966, -4.903451738719028),
#  (-4.146928971098447, -2.7933814828383854),
#  (-4.107682516774461, 2.850779602386939),
#  (1.6723758862824596, 4.71202280289272)]
obstacle_setting = [(2.1, 2.1), (-2.1, -2.1), (2.1, -2.1), (-2.1, 2.1),
                    (4.5, 0), (0, 4.5), (-4.5, 0), (0, -4.5),
                    ]
obstacle_setting = [(0, 2), (2, -2.1), (-2, -2.1),
                    (4.2, 2), (2.1, 4.7), (-2.1, 4.7), (-4.2, 2), (0, -5),
                    (5, -2.3), (-5, -2.3),
                    ]
obstacle_setting = [(-0, -0), (-2.75, 2.75), (-2.75, -2.75), (2.75,2.75), (2.75, -2.75)]
# obstacle_setting = [(-3, 0), (-1, 3.5), (-2, -2.5), (2.25, 0.5)]
obstacle_setting = [(-3, 3.5), (2.5, 2.5), (-2.5, -2), (3.5, -2.5)] # figure 5, example 1

obstacle_setting = [(-3.5, 4.5),
                    (-3, -0.), 
                    (-3.5, -4.5),
                    (0.0, 2.5),
                    (0.0, -2.5),
                    (3.5, 4.5), 
                    (3, 0.),
                    (3.5, -4.5), 
                    ] 
# obstacle_setting = [
#                     (0, 2), 
#                     (3.2, -2.1), 
#                     (-3.2, -2.1),
#                     (4.2, 2), 
#                     (2.1, 4.7), 
#                     (-2.1, 4.7), 
#                     (-4.2, 2), 
#                     (0, -5),
#                     ]
# obstacle_setting = [(-3.5, 3.5),
#                     (1.5, 3), 
#                     (-3.2, -2.5),
#                     (4.2, -0.5), 
#                     # (3.2, -2.1), 
# ]
# obstacle_setting =[
#     (-4, 0), 
#     (-0, 4), 
#     (0, -4),
#     (4, 0), 
# ]
obstacle_setting = [(-1.75, 2.1), (-3.5, -2), (-17.5, 21), (-35, -20)]
STATIC_OBSTACLE_RADII = [0.5]*len(obstacle_setting)
# obstacle_setting = [(0, 2.5), (3.5, -2.), (-3.5, -2.),
#                     (4.5, 4.5), (0, -5), (-4.5, 4.5),
#                                     ]
# STATIC_OBSTACLE_RADII = list(np.random.uniform(0.4, 0.8, len(STATIC_OBSTACLE_LOCATIONS)))
# STATIC_OBSTACLE_RADII = [0.5, 0.6, 0.4, 
#                             0.75, 0.6, 0.9,]

# Mij_general.POSITIONAL_SET_POINT = np.array([-5, 0])
# obstacle_setting = [(-5, 5), (3, 3)]
# Mij_general.POSITIONAL_SET_POINT = np.array([5, -5])
Mij_general.set_voronoi_partition(obstacle_setting, STATIC_OBSTACLE_RADII)
# Mij_general.MINIMUM_DISTANCE_OBSTACLES = 10
# Mij_general.vor_margin = 1.5
# Mij_general.radius_multiplier = 1.5
Mij_general.vor_margin_factor = 0.75


fign = 1
for index_lambda in range(len(obstacle_setting)):
    # plot voronoi boundaries as well
    plot_Mi(Mij_general, index_lambda, fign, res=51, POSITIONAL_SET_POINT=Mij_general.POSITIONAL_SET_POINT, bound=7,
            plot_text=False)#501, bound=7
    plt.savefig('plots/twoobstacle_M'+str(fign)+'.png')
    fign += 1