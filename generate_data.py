import numpy as np
from scipy.integrate import odeint
from skimage.draw import circle

def mass_spring_system(energy_low, energy_high, n_rollouts, span=50):
    trajectory_size = 300
    trajectories = np.zeros(shape=(n_rollouts, trajectory_size, 2))
    for i in range(n_rollouts):
        trajectory = []
        E = np.random.uniform(energy_low, energy_high)
        phi = np.random.uniform(0, np.pi)
        r = np.sqrt(2. * E)
        x = r * np.cos(phi)
        p = r * np.sin(phi)
        sol = odeint(mass_spring_diff_eq, y0=[x, p], t = np.linspace(0, span, trajectory_size))
        # adding random noise to the trajectory
        trajectories[i, ...] += sol + np.random.randn(*sol.shape) * 0.01
    return trajectories
        

def mass_spring_diff_eq(vec, t):
    x, p = vec
    return [p, -x]

class DataGenerator:
    DATASETS = {"mass_spring" : mass_spring_system}
    def __init__(self, dataset_name = "mass_spring", energy_range = [.2, 1.]):
        self.data_gen = self.DATASETS[dataset_name]
        self.energy_range = energy_range
    
    def get_dataset(self, n_rollouts=10, size = 64):
        
        trajectories = mass_spring_system(self.energy_range[0], self.energy_range[1], n_rollouts=n_rollouts)

        images = self._convert_to_images(trajectories, size)

        return images.astype(np.float32)
        
    def _convert_to_images(self, sequences, size, r = 5):

        X = np.zeros(shape=(sequences.shape[0], size, size, 3, sequences.shape[1]))

        for IND, coordinates in enumerate(sequences):
            y = coordinates[:, 0]
            x = size // 2
            y_min, y_max = np.min(y), np.max(y)
            for ind, curr_y  in enumerate(y):
                img = np.zeros(shape=(size, size, 3))
                y = int(((curr_y - y_min) / (y_max - y_min)) * size * 0.6 + 0.2 * size)
                rr, cc = circle(y, x, r)
                img[rr, cc, 0] = 1.
                X[IND, ..., ind] += img
        
        return X