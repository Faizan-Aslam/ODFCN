import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from get_coordinates import get_all_camera_coordinate, get_list_for_filter
from ellipse_plot import get_cov_ellipse
import pandas as pd


class ConstantTurnRate:
    def __init__(self):
        # Arbitrary initialisation of state
        self.sim_timestep = 0
        return
    
    def generate_next_sim_timestep(self):
        # Get the get next state
        self.sim_timestep += 1
        return

class PosSensor:
    def __init__(self):
        self.sim_timestep = 0
        return

    def get_measurement(self, m):
        # Measure position according to a measurement model
        m_pos = m[self.sim_timestep]
        self.sim_timestep += 1
        return m_pos

class KFEstimator:
    def __init__(self, init_est, init_cov):
        self.sim_timestep = 0
        self.state = init_est
        self.cov = init_cov
        return
    
    def get_estimate(self, measurement):
        # Wrap the estimate function so self is not modifiable by accident
        self.state, self.cov = KFEstimator.estimate(measurement, self.state, self.cov, self.sim_timestep)
        self.sim_timestep += 1
        return

    @staticmethod
    def estimate(measurement, state, covariance, sim_timestep):
         # Set model parameters
        S = 0.01
        w = 1.4 / 0.25
        T = 0.25
        sigma_p = np.sqrt(10)

        # Construct system matrix A and system noise covariance Q
        A = np.array(
            [
                [1, np.sin(w * T) / w, 0, -(1 - np.cos(w * T)) / w],
                [0, np.cos(w * T), 0, -np.sin(w * T)],
                [0, (1 - np.cos(w * T)) / w, 1, np.sin(w * T) / w],
                [0, np.sin(w * T), 0, np.cos(w * T)],
            ]
        )
        Q = S * np.array(
            [
                [T ** 3 / 3, T ** 2 / 2, 0, 0],
                [T ** 2 / 2, T, 0, 0],
                [0, 0, T ** 3 / 3, T ** 2 / 2],
                [0, 0, T ** 2 / 2, T],
            ]
        )
        # Construct measurement matrix H and measurement noise matrix R
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = sigma_p**2* np.eye(2)
        # Predict next state and covariance (prediction step)
        xp = A @ state
        Pp = A @ covariance @ A.T + Q
        # Compute Kalman gain
        #K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R)
        #nd_measurement = [0,0]
        for ind_measurement in measurement:
            # # Compute Kalman gain
            print(ind_measurement)
            # ind_measurement = [ind_measurement[i] + j[i] for i in range(len(ind_measurement))]

            #ind_measurement = [number / 4 for number in ind_measurement]

            K = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R)
            # Incorporate measurement into predicted state and covariance (update step)
            xu = xp + K @ (ind_measurement - H @ xp)
            Pu = (np.eye(4) - K @ H) @ Pp
            xp = xu
            Pp = Pu
        return xu, Pu
        raise NotImplementedError

class Simulation:
    def __init__(self, filter_state_init, filter_cov_init, sim_timesteps, no_of_camera_readings):
        # Save relevant variables
        self.gt = ConstantTurnRate()
        self.pos_sensor = PosSensor()
        self.kf = KFEstimator(filter_state_init, filter_cov_init)
        self.no_of_camera_readings =no_of_camera_readings
        self.list_measurements = get_all_camera_coordinate(f"C:/Users/faiza/Downloads/AMR_exercise_10_12_solution/AMR_exercise_10_12_solution/camera-{self.no_of_camera_readings}/")
        self.m = get_list_for_filter(self.list_measurements, self.no_of_camera_readings)
        # print('self.m is  :',self.m)
        self.max_sim_timesteps = sim_timesteps
        # print(self.max_sim_timesteps)
        return
    def run(self):
         # Create plot parameters
        fig = plt.figure()
        fig.set_size_inches(w=9, h=5)
        # KF plot
        pos_plot = fig.add_subplot(121)
        pos_plot.set_title("KF with Position Measurements")
        pos_plot.set_xlabel("x position")
        pos_plot.set_ylabel("y position")
        # measurements and covariances make the plots too busy, remove them after each time step
        to_remove = []
        # Uncertainities Plot
        uncer_plot = fig.add_subplot(122)
        uncer_plot.set_title("Uncertainities Measurements")
        uncer_plot.set_xlabel("Filter Steps")
        uncer_plot.set_ylabel("Uncertainity")
        # Create legend here as the gradual building of lines doesn't work nicely with legend()
        m_pos_line = Line2D(
            [0], [0], color="lightgrey", linewidth=1, linestyle="None", marker="x"
        )
        est_line = Line2D([606], [806], color="green", marker=".", linewidth=1)
        unce_line = Line2D([0], [0], color="orange", marker=".", linewidth=1)
        fig.legend(
            [ m_pos_line, est_line, unce_line],
            [
                "Position Measurement",
                "Estimate",
                "Uncertainities"
            ],
        )
        m_pos = np.array(self.pos_sensor.get_measurement(self.m))
        print('mpos is : ' , m_pos)
        to_remove.append(
        pos_plot.scatter([m_pos[0][0],m_pos[1][0]], [m_pos[0][1],m_pos[1][1]], marker="x", color="lightgrey")
        )
        # Estimates
        pos_plot.scatter(self.kf.state[0], self.kf.state[2], marker=".", color="green")
        # Covariances are plotted at 2 standard deviations
        ellipse = get_cov_ellipse(
            np.array([
                    [self.kf.cov[0][0], self.kf.cov[0][2]],
                    [self.kf.cov[2][0], self.kf.cov[2][2]],
                ]
            ),
            np.array([self.kf.state[0], self.kf.state[2]]),
            2,
            fill=False,
            linestyle="-",
            edgecolor="green",
        )
        pos_plot.add_artist(ellipse)
        to_remove.append(ellipse)
        # Estimates
        uncer_plot.scatter(self.kf.sim_timestep, np.average(self.kf.cov), marker=".", color="orange")
        prev_est_pos = self.kf.state
        prev_cov = np.average(self.kf.cov)
        plt.pause(0.5)
        # Plot positions, measurements and estimates at each time
        print('max time step :',self.max_sim_timesteps)

        uncertainity_list = []
        x_coordinate_estimate = []
        y_coordinate_estimate = []

        for _ in range(self.max_sim_timesteps):
          for p in to_remove:
                p.remove()
          to_remove = []
          # generate and measure
          self.gt.generate_next_sim_timestep()

          m_pos = np.array(self.pos_sensor.get_measurement(self.m))
          print('m_pos is in for loop ',m_pos)

          self.kf.get_estimate(m_pos)

          print('x mark in the graph : ', m_pos[0], m_pos[1])
          to_remove.append(pos_plot.scatter([m_pos[:,0]], [m_pos[:,1]], marker="x", color="lightgrey"))
          pos_plot.scatter(
                self.kf.state[0], self.kf.state[2], marker=".", color="green"
            )
          pos_plot.plot(
                [prev_est_pos[0], self.kf.state[0]],
                [prev_est_pos[2], self.kf.state[2]],
                linewidth=1,
                color="green",
            )
          uncer_plot.scatter(self.kf.sim_timestep, np.average(self.kf.cov), marker=".", color="orange")
          uncer_plot.plot(
                [_, self.kf.sim_timestep],
                [prev_cov, np.average(self.kf.cov)],
                linewidth=1,
                color="orange",
            )
          # Covariances are plotted at 2 standard deviations
          ellipse = get_cov_ellipse(
              np.array(
                  [
                      [self.kf.cov[0][0], self.kf.cov[0][2]],
                      [self.kf.cov[2][0], self.kf.cov[2][2]],
                  ]
              ),
              np.array([self.kf.state[0], self.kf.state[2]]),
              2,
              fill=False,
              linestyle="-",
              edgecolor="green",
          )
          # Save current as previous
          prev_est_pos = self.kf.state
          prev_cov = np.average(self.kf.cov)
          pos_plot.add_artist(ellipse)
          to_remove.append(ellipse)
          plt.pause(0.5)
          uncertainity_list.append(prev_cov)
          x_coordinate_estimate.append(prev_est_pos[0])
          y_coordinate_estimate.append(prev_est_pos[2])
          df = pd.DataFrame({'col': uncertainity_list, 'x_estimate' : x_coordinate_estimate, 'y_estimate' : y_coordinate_estimate })
          df.to_csv('_all_uncertainity.csv')
        plt.savefig('all_plot')


if __name__ == "__main__":
    filter_state_init = np.array([0,606,0,860])+np.random.multivariate_normal(np.zeros(4), 2+np.eye(4))
    filter_cov_init = 2+np.eye(4)
    sim = Simulation(filter_state_init, filter_cov_init, sim_timesteps=37, no_of_camera_readings=4)
    sim.run()