import numpy as np
from sklearn.cluster import KMeans

def generate_random_rod(center_point, mag):
    ndims = center_point.ndim+1
    displacement = np.random.uniform(-mag, mag, ndims)
    rod = [center_point + displacement, center_point - displacement]
    return rod

def find_critical_pointsV4(initial_points, state_difference, model, Env,
                          distthreshold=0.1, 
                          iterations=1, n_clusters=4, 
                          n_rodpoints=3, beta=0.5,
                          custom_state_init=None,
                          custom_state_to_observation=None,
                          check_bounds=None,
                          get_state_from_env=None, 
                          verbose=False,
                          env_kwargs=None):    
    SAMPLING_TIME_SECONDS = env_kwargs['SAMPLING_TIME_SECONDS']
    TIME_STEPS = env_kwargs['TIME_STEPS']

    # initialize the set of points to consider
    next_points = initial_points
    history = []
    rod_spacing = state_difference
    for iii in range(iterations):
        if state_difference <= SAMPLING_TIME_SECONDS:
            iii = iterations-1
        print("Iteration #", iii+1, " out of ", iterations)
        if len(next_points) >= 1:# n_clusters:
            print('rod spacing', rod_spacing )
            new_points = []
            if verbose:
                print('number of points: ', len(next_points))
            for center_point in next_points:                    
                    for n_rod in range(n_rodpoints):
                        # creating the rod 
                        start_points = generate_random_rod(center_point, rod_spacing)
                       
                        # simulate the system for both points in the rod
                        trajectories = []
                        length_shortest_trajectory = 10000
                        ignore_point = False
                        for start in start_points:
                            done = False
                            trajectory = []
                            # take one step at a time for each start point
                            # store end points after each TIME_STEPS
                            if custom_state_init is not None:
                                start = custom_state_init(start, model, env_kwargs['POSITIONAL_SET_POINT'])
                            env = Env(INITIAL_STATE=start, **env_kwargs)
                            env.reset()
                            if custom_state_to_observation is None:
                                obs = np.copy(start)
                            else:
                                obs = custom_state_to_observation(np.copy(start), env)
                            while done == False:
                                action, _ = model.predict(obs, deterministic=True)
                                obs, _, done, _,  _ = env.step(action)
                                if check_bounds is not None:
                                    if check_bounds(get_state_from_env(env), env):
                                        ignore_point = True
                                        break
                                        
                                if get_state_from_env is None:
                                    trajectory.append(env.state)
                                else:
                                    trajectory.append(get_state_from_env(env))
                            length_shortest_trajectory = min(len(trajectory), length_shortest_trajectory)
                            trajectories.append(np.array(trajectory))
                            
                        if ignore_point == False:
                            differences = []
                            initial_difference = np.linalg.norm(start_points[0]-start_points[1])
                            for step_traj in range(length_shortest_trajectory):
                                differences.append(max(
                                np.linalg.norm(trajectories[0][step_traj]-\
                                                trajectories[1][step_traj])-\
                                     initial_difference, 0) * env.SAMPLING_TIME_SECONDS)
                            difference = np.sum(differences)
                            
                            if difference > distthreshold:
                            # the new points for the next loop are taken slightly
                            # spaced from the orignal point
                            # dont generate new points for the last iteration
                                print('difference : ', difference, 'center point :', center_point)
                                if iii == iterations-1:
                                    new_points.append(center_point)
                                else:
                                    #new_rod = generate_random_rod(center_point, rod_spacing) 
                                    #new_points.extend(new_rod)
                                    new_points.extend(start_points)
                                    new_points.append(center_point)
                                break
        # NEW: check whether new points are in the state space
        valid_points = []
        if check_bounds is not None:
            for index, new_point in enumerate(new_points):    
                if check_bounds(new_point, env) == False: 
                    valid_points.append(new_point)
        else:
            valid_points = new_points
            # incase there are many points, we can 'summarize' clouds of points by clustering
        if iii < iterations-1:
            if len(valid_points)>n_clusters:
                cluster_array = np.array(valid_points)
                if center_point.ndim+1 == 1:
                    cluster_array = cluster_array.reshape(-1,1)
                kmeans = KMeans(n_clusters=min(n_clusters,len(valid_points)),
                                random_state=0).fit(cluster_array)
                cluster_centers = kmeans.cluster_centers_
                if center_point.ndim+1 == 1:
                    cluster_centers = cluster_centers[0]
                valid_points = cluster_centers
            next_points = []
            if check_bounds is not None:
                for index, valid_point in enumerate(valid_points):    
                    if check_bounds(valid_point, env) == False: 
                        next_points.append(valid_point)
            else:
                next_points = valid_points
            history.append(valid_points)
            rod_spacing = beta*rod_spacing
            if rod_spacing < SAMPLING_TIME_SECONDS:
                SAMPLING_TIME_SECONDS = SAMPLING_TIME_SECONDS*beta
                TIME_STEPS = int(TIME_STEPS*1/beta)
                env_kwargs['SAMPLING_TIME_SECONDS'] = max(SAMPLING_TIME_SECONDS, 0.05)
                env_kwargs['TIME_STEPS'] = min(TIME_STEPS, 100)
                print('number of TIME_STEPS ', TIME_STEPS)
    return valid_points, history

class Vor_part:
    def __init__(self, STATIC_OBSTACLE_LOCATIONS, STATIC_OBSTACLE_RADII=0.5):
        self.points = STATIC_OBSTACLE_LOCATIONS
        if type(STATIC_OBSTACLE_RADII) == list:
            self.STATIC_OBSTACLE_RADII = STATIC_OBSTACLE_RADII
        else:
            self.STATIC_OBSTACLE_RADII = [STATIC_OBSTACLE_RADII] * (len(STATIC_OBSTACLE_LOCATIONS))

class Def_Mob_general:
    # SVM approach general
    def __init__(self, svm_model, POSITIONAL_SET_POINT, reference_obstacle, vor_margin_factor=0.75, WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE=2):
        self.svm_model = svm_model
        self.POSITIONAL_SET_POINT = POSITIONAL_SET_POINT
        self.reference_obstacle = reference_obstacle
        self.set_voronoi_partition([reference_obstacle])
        self.vor_margin_factor = vor_margin_factor
        self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE = WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE
    
    def set_voronoi_partition(self, STATIC_OBSTACLE_LOCATIONS, STATIC_OBSTACLE_RADII=0.5):
        self.vor = Vor_part(np.array(STATIC_OBSTACLE_LOCATIONS), STATIC_OBSTACLE_RADII)

    def check_voronoi(self, point, index_lambda, res=50):
        x, y = point[0], point[1]
        nearest_point_index = np.argmin(np.sum((self.vor.points - point)**2, axis=1)) 
        if nearest_point_index == index_lambda:
            return True
        distance_to_nearest_obstacle = np.linalg.norm(self.vor.points[nearest_point_index]-point)
        if distance_to_nearest_obstacle - self.vor.STATIC_OBSTACLE_RADII[nearest_point_index] <= np.sqrt(2)*self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2:
            return False
        alphas = np.linspace(0,2*np.pi, res, endpoint=False)
        in_voronoi_boundary = []
        for alpha in alphas:
            near_point = np.array([x, y]) + \
                self.vor_margin_factor*distance_to_nearest_obstacle*np.array([np.cos(alpha), np.sin(alpha)])
            point_index = np.argmin(np.sum((self.vor.points - near_point)**2,
                                            axis=1))
            
            in_voronoi_boundary.append(point_index==index_lambda)
        
        if sum(in_voronoi_boundary) > 0:
            return True
        return False

    def cart2pol(self, z):
        x, y = z[0], z[1]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def pol2cart(self, z):
        rho, phi = z[0], z[1]
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)

    def transform_coordinates(self, position_to_transform, targeted_obstacle):
        polar_position_to_transform = self.cart2pol(position_to_transform - self.POSITIONAL_SET_POINT)
        targeted_obstacle_setpoint_adjusted = targeted_obstacle - self.POSITIONAL_SET_POINT
        polar_target_obstacle = self.cart2pol(targeted_obstacle_setpoint_adjusted)
        polar_reference_obstacle = self.cart2pol(self.reference_obstacle)
        scale_to_transform = polar_target_obstacle[0] / polar_reference_obstacle[0]
        angle_to_transform = polar_target_obstacle[1] - polar_reference_obstacle[1]
        transformed_polar = (polar_position_to_transform[0]/scale_to_transform, polar_position_to_transform[1]-angle_to_transform)
        transformed_cartesian = self.pol2cart(transformed_polar)
        return transformed_cartesian

    def predict_set_svm(self, point,  index_lambda):
        obstacle_targeted = self.vor.points[index_lambda]
        translated_point = self.transform_coordinates(point, obstacle_targeted)
        predicted_set_svm = self.svm_model.predict(np.array([translated_point]))
        return predicted_set_svm
    
    def check_in_Mi(self, point, index_q, index_lambda):
        predicted_set_svm = self.predict_set_svm(point, index_lambda)
        if predicted_set_svm == 2:
            # inside the intersection of M_ext^0 and M_ext^1
            return True
        elif predicted_set_svm == index_q:
            # inside M_ext^j
            return True
        else:
            return False
        
    def get_in_Mi(self, point, index_q, index_lambda=0):
        if self.check_voronoi(point, index_lambda):
            return self.check_in_Mi(point, index_q, index_lambda)
        else:
            return False
    
    def near_boundary(self, point, index_q, index_lambda, boundary_margin=0.3, RESOLUTION_SEARCH=10):
        alphas = np.linspace(0,2*np.pi, num=RESOLUTION_SEARCH, endpoint=False)
        in_Mij = []
        for alpha in alphas:
            near_point = point + np.array([np.cos(alpha), np.sin(alpha)]) * boundary_margin
            in_Mij.append(self.get_in_Mi(near_point, index_q, index_lambda))

        in_Mij = np.array(in_Mij)
        if np.sum(in_Mij) > 0 and np.prod(in_Mij) == 0:
            return True
        else:
            return False

    
    def distance_to_boundary_hyrl(self, point, index_q, index_lambda, WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE=2, RESOLUTION_SEARCH=5):
        # check if point is inside Mij
        if self.get_in_Mi(point, index_q, index_lambda):
            return WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2 + 1 # setting distance larger than observaiton window
        
        # check if point is in observable range:
        alphas = np.linspace(0,2*np.pi, num=RESOLUTION_SEARCH, endpoint=False)
        min_distance_to_boundary = WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2
        near_boundary = False
        for alpha in alphas:
            near_point = point + np.array([np.cos(alpha), np.sin(alpha)]) * min_distance_to_boundary
            if self.get_in_Mi(near_point, index_q, index_lambda):
                near_boundary = True
                break
        # if we are not near the boundary for any of the evaluated points above, return maximal observable range
        if near_boundary  == False:
            return min_distance_to_boundary
        
        # we are in obserable range to the boundary, now we want to find the distance to the boundary via binary search
        low_range, max_range = 0, min_distance_to_boundary
        for _ in range(RESOLUTION_SEARCH):
            mid_range = (low_range + max_range) / 2
            near_boundary = False
            for alpha in alphas:
                near_point = point + np.array([np.cos(alpha), np.sin(alpha)]) * mid_range
                if self.get_in_Mi(near_point, index_q, index_lambda):
                    near_boundary = True
                    max_range = mid_range
                    break
            if near_boundary == False:
                low_range = mid_range
        return mid_range
