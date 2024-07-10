import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib
import pickle
from scipy.spatial import ConvexHull
from sklearn import svm 
from env_static_obstacles import BirdsEyeViewStaticObstacleLocationsEnvironment
from stable_baselines3 import PPO
from utils import find_critical_pointsV4, Def_Mob_general

def contained_in_hull(point, hull):
    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    eps = np.finfo(np.float32).eps
    return np.all(np.asarray(point) @ A.T + b.T < eps, axis=1)


def plot_Mij(Mij, j, env, exthull, poly, polies, 
            color='lightblue', fign=1, res=100):
    # j is either 0 or 1
    xp = np.linspace(env.MINIMAL_VALUE_POSITION_X/1., env.MAXIMAL_VALUE_POSITION_X/1., res)
    yp = np.linspace(env.MINIMAL_VALUE_POSITION_Y/1., env.MAXIMAL_VALUE_POSITION_Y/1., res)
    xi,yi = np.meshgrid(xp,yp)
    in_Mij = np.zeros((res,res))
    for idx in range(res):
        for idy in range(res):
            point = (xp[idx], yp[idy])
            in_Mij[idy, idx] = Mij.get_in_Mi(point, j, Mi_margin=0)
    plt.figure(fign)
    cmap = clrs.ListedColormap(['none', color])
    plt.scatter(xi, yi, s=15, c=in_Mij, rasterized=True, cmap =cmap)
    plt.plot(POSITIONAL_SET_POINT[0], POSITIONAL_SET_POINT[1], 'o', color='red')
    for id_ob, obstacle in enumerate(env.STATIC_OBSTACLE_LOCATIONS):
        x_obst, y_obst = obstacle[0], obstacle[1]
        obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                        radius=env.STATIC_OBSTACLE_RADII[id_ob], color='gray')
        plt.gca().add_patch(obstaclePatch)
    plt.grid(visible=True)
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    for simplex in exthull.simplices:
        plt.plot(exthull.points[simplex, 0], exthull.points[simplex, 1], 
                  '-', color='red', linewidth=2) 
    xp = np.linspace(env.MINIMAL_VALUE_POSITION_X, env.MAXIMAL_VALUE_POSITION_X)
    plt.plot(xp, poly(xp), color='red')
    alpha = np.linspace(0,1, res)
    polies0, polies1 = polies[0], polies[1]
    polypoints_fitted0 = np.vstack( ply(alpha) for ply in polies0 ).T
    polypoints_fitted1 = np.vstack( ply(alpha) for ply in polies1 ).T
    plt.plot(*polypoints_fitted0.T, '--', color='blue')
    plt.plot(*polypoints_fitted1.T, '--', color='blue')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.tight_layout()
    
def plot_Mjs(Mij, env, fign=1, res=100):
    # j is either 0 or 1
    xp = np.linspace(env.MINIMAL_VALUE_POSITION_X/1., env.MAXIMAL_VALUE_POSITION_X/1., res)
    yp = np.linspace(env.MINIMAL_VALUE_POSITION_Y/1., env.MAXIMAL_VALUE_POSITION_Y/1., res)
    xi,yi = np.meshgrid(xp,yp)
    in_Mi0 = np.zeros((res,res))
    in_Mi1 = np.zeros((res,res))
    in_Mi01 = np.zeros((res,res))
    for idx in range(res):
        for idy in range(res):
            point = (xp[idx], yp[idy])
            in_Mijs0, in_Mijs1 = 0, 0
            for Mii in Mij:
                in_Mijs0 += Mii.get_in_Mi(point, 0, Mi_margin=Mii.vor_margin)
                in_Mijs1 += Mii.get_in_Mi(point, 1, Mi_margin=Mii.vor_margin)
            in_Mi0[idy, idx] = np.sum(in_Mijs0) >= 1 
            in_Mi1[idy, idx] = np.sum(in_Mijs1) >= 1 
            in_Mi01[idy, idx] = in_Mi0[idy, idx] and in_Mi1[idy, idx]
    plt.figure(fign)
    cmap = clrs.ListedColormap(['none', 'lightblue'])
    plt.scatter(xi, yi, s=15, c=in_Mi0, rasterized=True, cmap =cmap)
    cmap = clrs.ListedColormap(['none', 'lightcoral'])
    plt.scatter(xi, yi, s=15, c=in_Mi01, rasterized=True, cmap =cmap)
    plt.plot(POSITIONAL_SET_POINT[0], POSITIONAL_SET_POINT[1], 'o', color='red')
    for id_ob, obstacle in enumerate(env.STATIC_OBSTACLE_LOCATIONS):
        x_obst, y_obst = obstacle[0], obstacle[1]
        obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                        radius=env.STATIC_OBSTACLE_RADII[id_ob], color='gray')
        plt.gca().add_patch(obstaclePatch)
    plt.grid(visible=True)
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.tight_layout()
    
    plt.figure(fign+1)
    cmap = clrs.ListedColormap(['none', 'lightgreen'])
    plt.scatter(xi, yi, s=15, c=in_Mi1, rasterized=True, cmap =cmap)
    cmap = clrs.ListedColormap(['none', 'lightcoral'])
    plt.scatter(xi, yi, s=15, c=in_Mi01, rasterized=True, cmap =cmap)
    plt.plot(POSITIONAL_SET_POINT[0], POSITIONAL_SET_POINT[1], 'o', color='red')
    for id_ob, obstacle in enumerate(env.STATIC_OBSTACLE_LOCATIONS):
        x_obst, y_obst = obstacle[0], obstacle[1]
        obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                        radius=env.STATIC_OBSTACLE_RADII[id_ob], color='gray')
        plt.gca().add_patch(obstaclePatch)
    plt.grid(visible=True)
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.tight_layout()
    
def plot_Mi(Mij, env, exthull, poly,
            fign=1, res=100):
    # j is either 0 or 1
    xp = np.linspace(env.MINIMAL_VALUE_POSITION_X/1., env.MAXIMAL_VALUE_POSITION_X/1., res)
    yp = np.linspace(env.MINIMAL_VALUE_POSITION_Y/1., env.MAXIMAL_VALUE_POSITION_Y/1., res)
    xi,yi = np.meshgrid(xp,yp)
    in_Mi0 = np.zeros((res,res))
    in_Mi1 = np.zeros((res,res))
    for idx in range(res):
        for idy in range(res):
            point = (xp[idx], yp[idy])
            in_Mi0[idy, idx] = Mij.get_in_Mi(point, 0, Mi_margin=0) 
            in_Mi1[idy, idx] = Mij.get_in_Mi(point, 1, Mi_margin=0)
    in_Mi01 = in_Mi0*in_Mi1
    plt.figure(fign)
    cmap = clrs.ListedColormap(['none', 'lightblue'])
    plt.scatter(xi, yi, s=15, c=in_Mi0, rasterized=True, cmap =cmap)
    cmap = clrs.ListedColormap(['none', 'lightgreen'])
    plt.scatter(xi, yi, s=15, c=in_Mi1, rasterized=True, cmap =cmap)
    cmap = clrs.ListedColormap(['none', 'lightcoral'])
    plt.scatter(xi, yi, s=15, c=in_Mi01, rasterized=True, cmap =cmap)
    plt.plot(POSITIONAL_SET_POINT[0], POSITIONAL_SET_POINT[1], 'o', color='red')
    for id_ob in range(env.NUMBER_OF_STATIC_OBSTACLES):
        obstacle = env.STATIC_OBSTACLE_LOCATIONS[id_ob]
        x_obst, y_obst = obstacle[0], obstacle[1]
        obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                        radius=env.STATIC_OBSTACLE_RADII[id_ob], color='gray')
        plt.gca().add_patch(obstaclePatch)
    plt.grid(visible=True)
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.tight_layout()
    
    
class CHull(ConvexHull):
    def __init__(self, points, incremental=False):
        ConvexHull.__init__(self, points, incremental)
    def centrum(self):
        c = []
        for i in range(self.points.shape[1]):
            c.append(np.mean(self.points[self.vertices,i]))
        return c

def state_to_observation(state, env):
    x, y = state[0], state[1]
    xi = env.orientation_vector_vehicle
    return env.update_observation(x,y, xi)

def custom_state_init(start, model, POSITIONAL_SET_POINT, env=None, res=50):
    theta = np.arctan2(POSITIONAL_SET_POINT[1]-start[1], POSITIONAL_SET_POINT[0]-start[0])
    return np.array([start[0], start[1], np.cos(theta), np.sin(theta)])


def get_state_from_env(env):
    return np.array([env.horizontal_position_vehicle, env.vertical_position_vehicle], dtype=np.float32)

def check_bounds(point, env):
    x, y = point[0], point[1]
    env.x, env.y = x, y
    env.check_bounds(x,y,env.orientation_vector_vehicle)
    return env.out_of_bounds

def plot_line(pf, color='red', linestyle='--'):
    a, b = pf[0], pf[1]
    x = np.array([0,4])
    y = np.array([a*x[0]+b, a*x[1]+b])
    plt.plot(x, y, color=color, linestyle=linestyle)
    return x, y

def get_coeffs(xs, ys):
    return np.polyfit(xs, ys, 1)
    
def find_intersection(coefs1, coefs2):
    a1, b1 = coefs1[0], coefs1[1]
    a2, b2 = coefs2[0], coefs2[1]
    if abs(a1-a2) <= 1e-6:
        intersection = False
        pI = None
    else:
        intersection = True
        xI = (b2-b1)/(a1-a2)
        yI = a1*xI+b1
        pI = np.array([xI, yI])
    return intersection, pI

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.close('all')

t_sampling=0.05
steps = 100

np.random.seed(1)
index = 0
save = True
load_points = True
animate = False
np.random.seed(1)
only_find_critical_points = False 
number_of_datapoints = 1
ex_number = 11
NUMBER_OF_STATIC_OBSTACLES = 1

if ex_number == 11:
    trained_model = PPO.load("agents/ppo_Dubin_static_futuresteps0_example11_v4.zip")
    obstacles = [(-2, 0)]
    STATIC_OBSTACLE_RADII = [0.5]

POSITIONAL_SET_POINT = np.array([0.,0])

save_name = 'Mstars/ob1_Mstar_x'+str(POSITIONAL_SET_POINT[0])+'_y'+str(POSITIONAL_SET_POINT[1])+'_static_example'+str(ex_number)+'_v1'
save_name = save_name.replace('.','')
save_name = save_name.replace('-','m')
save_name = save_name+'.pkl'

kwargs = dict(
                initialize_state_randomly=False, 
                TIME_STEPS=steps, 
                train=True, 
                POSITIONAL_SET_POINT=POSITIONAL_SET_POINT, 
                NUMBER_OF_STATIC_OBSTACLES=NUMBER_OF_STATIC_OBSTACLES, 
                RANDOM_NUMBER_OF_STATIC_OBSTACLES=False,
                clip_through_obstacles=True,
                SAMPLING_TIME_SECONDS=t_sampling, 
                STATIC_OBSTACLE_LOCATIONS=obstacles, 
                STATIC_OBSTACLE_RADII=STATIC_OBSTACLE_RADII,
                VEHICLE_MODEL='Dubin',
                IGNORE_OBSTACLES=False
)
env = BirdsEyeViewStaticObstacleLocationsEnvironment(**kwargs)



if load_points == False:
    resolution = 25
    if ex_number == 11:
        resolution = 20
        x_ = np.linspace(-5, -2.5, int(resolution*1))
        y_ = np.linspace(-1, 1, resolution)
        n_clusters = 50
    state_difference = LA.norm(np.array([x_[1]-x_[0], y_[1]-y_[0]]))
    initial_points = []
    
    for idx in range(int(round(resolution*1))):
        for idy in range(resolution):
            init_cond = np.array([x_[idx], y_[idy]], dtype=np.float32)
            initial_points.append(init_cond)
            
    M_star, history = find_critical_pointsV4(initial_points, state_difference, trained_model, 
                                    BirdsEyeViewStaticObstacleLocationsEnvironment, distthreshold=5,
                                    iterations=20, 
                                    n_clusters = n_clusters, beta = 0.8,
                                    custom_state_to_observation=state_to_observation,
                                    get_state_from_env=get_state_from_env,
                                    custom_state_init=custom_state_init,
                                    check_bounds=check_bounds, verbose=True,
                                    env_kwargs=kwargs,
                                    n_rodpoints=4,
                                    )
    # try plotting all the critical points before clustering

    M_star = np.array(M_star)
    if save==True:
        with open(save_name, 'wb') as outp:
            pickle.dump(M_star, outp, -1)
else:
    with open(save_name, 'rb') as inp:
        M_star= pickle.load(inp)  
    
xc, yc = M_star.T
colors = ['red', 'green', 'blue', 'black']

cc = 0
xc, yc = M_star.T
if load_points == False:
    plt.close('all')
    
    number = 0
    for itt in history:
        plt.plot(env.HORIZONTAL_POSITIONAL_SET_POINT, env.VERTICAL_POSITIONAL_SET_POINT, 'o', color='red')
        plt.text(env.HORIZONTAL_POSITIONAL_SET_POINT+0.1,env.VERTICAL_POSITIONAL_SET_POINT+.4, r'$p^*$', fontsize=22, color='red')
        for idob, obstacle in enumerate(env.STATIC_OBSTACLE_LOCATIONS):
            x_obst, y_obst = obstacle[0], obstacle[1]
            obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                            radius=env.STATIC_OBSTACLE_RADII[idob], color='gray')
            plt.gca().add_patch(obstaclePatch)
        plt.grid(visible=True)
        plt.xlabel('$p_x$', fontsize=22)
        plt.ylabel('$p_y$', fontsize=22)
        
        plt.tight_layout()
        xc, yc = np.array(itt).T
        if cc >= len(colors):
            cc = 0
        plt.plot(xc[:-1], yc[:-1], 'x', color=colors[cc], markersize=12)
        hull = CHull(np.array(itt))
        plt.text(-1, 3.5, f'Iteration {number+1}', fontsize=22)
        if animate:
            plt.savefig('gifstorage/multi1_ob_critpoints_'+f'_n{number}'+'_dubin.png', 
                                transparent = False,  
                                facecolor = 'white'
                                )
        cc += 1
        number += 1
        plt.pause(1)
        plt.clf()
    
    # if animate:
    #     frames = []
    #     for idx in range(number):
    #         image = imageio.v2.imread('gifstorage/multi'+str(n_obs)+'_ob_critpoints_'+f'_n{idx}'+'_dubin.png')
    #         frames.append(image)    
    #     imageio.mimsave('gifs/example'+str(ex_number)+'+_critpoints'+'.gif', # output gif
    #                     frames,          # arr00ay00 of input frames
    #                     fps = 2)         # optional: frames per second      
plt.close('all')
plt.plot(xc, yc, 'x', color='red', markersize=2)
savefig_name = save_name.replace('.pkl','.png')
savefig_name = savefig_name.replace('Mstars', 'plots')
plt.savefig(savefig_name)

if only_find_critical_points == False:
    ccc = 5

    obstacle = env.STATIC_OBSTACLE_LOCATIONS[0]
    radius_obstacle = STATIC_OBSTACLE_RADII[0]
    xi, yi = M_star.T
    hull = CHull(M_star)
    c = hull.centrum()
    
            
    xc = [c[0], POSITIONAL_SET_POINT[0], obstacle[0]]
    yc = [c[1], POSITIONAL_SET_POINT[1], obstacle[1]]
    weights = [1, 1, 1]
    poly = np.polynomial.polynomial.Polynomial(np.flip(np.polyfit(xc, yc, 2)))

    rho = 0.5
    ext_boundary_points = []
            
    # Extending the hull
    for simplex in hull.simplices:
        for ii in range(2):
            boundary_point = M_star[simplex][ii]
            alpha = np.arctan2(boundary_point[1]-c[1], 
                                boundary_point[0]-c[0])
            for theta in np.linspace(0,2*np.pi,100, endpoint=False):
                ext_boundary_point = np.array([boundary_point[0]+rho*np.cos(theta),
                                    boundary_point[1]+rho*np.sin(theta)])
                ext_boundary_points.append(ext_boundary_point)

    exthull_circle = ConvexHull(ext_boundary_points)

    d_spob = min(2, # 2
                env.get_distance_obs(obstacle, radius_obstacle, env.HORIZONTAL_POSITIONAL_SET_POINT, env.VERTICAL_POSITIONAL_SET_POINT)
                    )
    
    plt.figure(ccc)
    ccc += 1
    
    for simplex in exthull_circle.simplices:
        plt.plot(exthull_circle.points[simplex, 0], exthull_circle.points[simplex, 1], 
                  '--', color='red', linewidth=2) 

    xp = np.linspace(env.MINIMAL_VALUE_POSITION_X, env.MAXIMAL_VALUE_POSITION_X, 100)
    
    
    plt.plot(xp, poly(xp)+0.2, color='red', linestyle='--')
    plt.plot(xp, poly(xp)-0.2, color='red', linestyle='--')
    alphas_plot = np.linspace(0, 2*np.pi, 100, endpoint=False )
    for id_ob in range(env.NUMBER_OF_STATIC_OBSTACLES):
        obstacle = env.STATIC_OBSTACLE_LOCATIONS[id_ob]
        x_obst, y_obst = obstacle[0], obstacle[1]
        obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                        radius=env.STATIC_OBSTACLE_RADII[id_ob], color='gray',zorder=10)
        plt.gca().add_patch(obstaclePatch)
    plt.plot(xp, poly(xp), color='black',zorder=11)
    plt.plot(np.cos(alphas_plot)*d_spob, np.sin(alphas_plot)*d_spob, color='red', linestyle='--')
    plt.grid(visible=True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    plt.text(3, 0.5, r'$y_c$', fontsize=22, color='black')
    plt.tight_layout()


    resolution_grid_svm = 300
    # 0 if in M0
    # 1 if in M1
    # 2 if in crit points / near target
    x_pos_SVM, y_pos_SVM = np.linspace(-6, 6, resolution_grid_svm), np.linspace(-6, 6, resolution_grid_svm)
    LABELS_SVM = []
    DATA_POINTS_SVM = []
    SVM_margin = 0.2
    sign = 1
    if c[0] < 0:
        sign = -1
    for iix in range(resolution_grid_svm):
        for iiy in range(resolution_grid_svm):
            point_to_evaluate =  (x_pos_SVM[iix], y_pos_SVM[iiy])
            DATA_POINTS_SVM.append(point_to_evaluate)
            if contained_in_hull(point_to_evaluate, exthull_circle) \
                or np.linalg.norm(np.array(point_to_evaluate)-POSITIONAL_SET_POINT) <= d_spob \
                    or (poly(point_to_evaluate[0]) < point_to_evaluate[1] + SVM_margin and poly(point_to_evaluate[0]) > point_to_evaluate[1] - SVM_margin):
                LABELS_SVM.append(2)
            elif sign*poly(point_to_evaluate[0]) > sign*point_to_evaluate[1]:
                LABELS_SVM.append(0)
            else:
                LABELS_SVM.append(1)
    xi_SVM, yi_SVM = np.meshgrid(x_pos_SVM, y_pos_SVM)
    cmap = clrs.ListedColormap(['lightblue', 'lightgreen', 'lightcoral'])
    plt.figure(ccc, figsize=(6.4, 3))
    plt.scatter(yi_SVM, xi_SVM, s=15, c=LABELS_SVM, rasterized=True, cmap=cmap)

    plt.plot(xp, poly(xp)+0.2, color='red', linestyle='--')
    plt.plot(xp, poly(xp)-0.2, color='red', linestyle='--')
    plt.plot(np.cos(alphas_plot)*d_spob, np.sin(alphas_plot)*d_spob, color='red', linestyle='--')
    for simplex in exthull_circle.simplices:
        plt.plot(exthull_circle.points[simplex, 0], exthull_circle.points[simplex, 1], 
                  '--', color='red', linewidth=2) 
    
    plt.grid(visible=True)
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    

    svm_model = svm.SVC(kernel='rbf', gamma='auto', class_weight={0:1, 1:1, 2:1})
    svm_model.fit(DATA_POINTS_SVM, LABELS_SVM)


    Z = svm_model.predict(np.c_[xi_SVM.ravel(), yi_SVM.ravel()])

    Z = Z.reshape(xi_SVM.shape)
    cmap2 = clrs.ListedColormap(['None', 'None', 'lightcoral'])
    for id_ob in range(env.NUMBER_OF_STATIC_OBSTACLES):
        obstacle = env.STATIC_OBSTACLE_LOCATIONS[id_ob]
        x_obst, y_obst = obstacle[0], obstacle[1]
        obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                        radius=env.STATIC_OBSTACLE_RADII[id_ob], color='gray', zorder=10)
        plt.gca().add_patch(obstaclePatch)
    plt.plot(0, 0, '*', color='red', markersize=12, zorder=12)
    plt.xlim(-6, 3)
    plt.ylim(-2., 2.)
    plt.tight_layout()

    resplot=50
    fign = ccc + 1
    Mij = Def_Mob_general(svm_model, POSITIONAL_SET_POINT, obstacle, vor_margin_factor=0.75)

    plt.savefig('plots/example'+str(ex_number)+'_Mgeneral.pdf')

    save_name = save_name.replace('Mstar', 'Mij')
    save_name = save_name.replace('Mijs','hyrl_sets')
    with open(save_name, 'wb') as outp:
        pickle.dump(Mij, outp, -1)
        print(' saved as ', save_name)
    