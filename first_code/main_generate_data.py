import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat 

from create_object import create_object
from compute_normal import compute_normal
from plot_scene_dual_training_selfOcc import plot_scene_dual_training_selfOcc
from plane_line_intersect import plane_line_intersect

# Initialize parameters
xmin = -3/2
xmax = 3/2
ymax = 3
zmax = 3
params = {
    'camera_FOV_center': [0, -0.25 / 2, 0],
    'camera_FOV': 0.25,
    'cam_pixel_dim': 32,
    'bin_size': 390e-12,
    'FOV_radius': 0.25/32,
    'c': 299792458,
    'furthest_scene_point': [xmax, ymax, zmax],
    'furthest_spad_point': [-0.25/2, -0.25, 0],
    'laser_intensity': 1000,
    'laser_pos': np.array([0, 0, 0]),
    'laser_normal': np.array([0, 0, 1]),
    'wall_discr': 299792458 / 2 * 390e-12 / 4,
    'num_time_bins': 154,
}

# Background objects
objects = []

# back wall (i=1)
objects.append(create_object(xmin, ymax, 0, xmax - xmin, zmax, 0, 'wall', 'back'))

# right side wall (i=2)
objects.append(create_object(xmax, ymax, 0, ymax, zmax, -np.pi / 2, 'wall', 'back'))

# left side wall (i=3)
objects.append(create_object(xmin, ymax, 0, ymax, zmax, -np.pi / 2, 'wall', 'back'))

# ceiling (i=4)
i = len(objects)
objects.append([None] * 5)
objects[i][0] = 'wall'
objects[i][1] = np.array([xmin, ymax, zmax])
objects[i][2] = np.array([xmax - xmin, 0, 0])
objects[i][3] = np.array([0, -ymax, 0])
objects = compute_normal(objects, i)

# Now, include this in your main data generation function
def main_generate_data(xcoord, ycoord):
    params['xcoord'] = xcoord
    params['ycoord'] = ycoord

    for xx in range(len(xcoord)):
        # Reset objects list for each coordinate pair
        objects = []

        # Add background objects
        objects.append(create_object(xmin, ymax, 0, xmax - xmin, zmax, 0, 'wall', 'back'))
        objects.append(create_object(xmax, ymax, 0, ymax, zmax, -np.pi / 2, 'wall', 'back'))
        objects.append(create_object(xmin, ymax, 0, ymax, zmax, -np.pi / 2, 'wall', 'back'))
        i = len(objects)
        objects.append([None] * 5)
        objects[i][0] = 'wall'
        objects[i][1] = np.array([xmin, ymax, zmax])
        objects[i][2] = np.array([xmax - xmin, 0, 0])
        objects[i][3] = np.array([0, -ymax, 0])
        objects = compute_normal(objects, i)

        # Add other scene objects and processing here
        # ...

        # Plot the scene
        plt.figure()
        plot_scene_dual_training_selfOcc(params['laser_pos'], objects, params)
        plotFlag = 0

        # Discretize the scene into patches
        numpatches = 0
        for o in range(len(objects)):
            numpatches += round(np.linalg.norm(objects[o][2]) / params['wall_discr']) * round(np.linalg.norm(objects[o][3]) / params['wall_discr'])

        # Pre-allocate matrices
        scene_pixel = np.zeros((int(np.ceil(numpatches)), 3))
        scene_pixel_normal = np.zeros((int(np.ceil(numpatches)), 3))
        scene_pixel_area = (params['wall_discr'])**2
        scene_pixel_angle = 1000 * np.ones(int(np.ceil(numpatches)))

        v = nmz(params['laser_pos'][:2])
        light_angle = v[0]
        patch_count = 1

        for o in range(len(objects)):
            for i in np.linspace(0, 1, round(np.linalg.norm(objects[o][2]) / params['wall_discr'])):
                for j in np.linspace(0, 1, round(np.linalg.norm(objects[o][3]) / params['wall_discr'])):
                    if objects[o][1][2] + i * objects[o][2][2] + j * objects[o][3][2] > 0:
                        pos = objects[o][1] + i * objects[o][2] + j * objects[o][3]
                        oc = False

                        if o <= 4:
                            for k in range(5, len(objects)):
                                I, check = plane_line_intersect(objects[k][4], objects[k][1], params['laser_pos'], pos)
                                A = objects[k][1]
                                B = objects[k][1] + objects[k][2]
                                C = objects[k][1] + objects[k][3]
                                D = objects[k][1] + objects[k][2] + objects[k][3]
                                quad = np.array([A, B, D, C])

                                if check == 1:
                                    mi = np.min(quad, axis=0)
                                    ma = np.max(quad, axis=0)
                                    T = (I[0] >= mi[0]) and (I[0] <= ma[0]) and \
                                        (I[1] >= mi[1]) and (I[1] <= ma[1]) and \
                                        (I[2] >= mi[2]) and (I[2] <= ma[2])

                                    if T:
                                        oc = True
                                        break

                        v = nmz(pos[:2])
                        scene_pixel_angle = -v[0]
                        if -scene_pixel_angle + light_angle < 0:
                            oc = True

                        if not oc:
                            scene_pixel[patch_count, :] = pos
                            scene_pixel_normal[patch_count, :] = objects[o][4]
                            patch_count += 1

        # Find the centers of the camera/floor pixels
        pixel_x = np.linspace(params['camera_FOV_center'][0] - params['camera_FOV'] / 2 + params['camera_FOV'] / (2 * params['cam_pixel_dim']),
                              params['camera_FOV_center'][0] + params['camera_FOV'] / 2 - params['camera_FOV'] / (2 * params['cam_pixel_dim']), params['cam_pixel_dim'])
        pixel_y = np.linspace(params['camera_FOV_center'][1] - params['camera_FOV'] / 2 + params['camera_FOV'] / (2 * params['cam_pixel_dim']),
                                 params['camera_FOV_center'][1] + params['camera_FOV'] / 2 - params['camera_FOV'] / (2 * params['cam_pixel_dim']), params['cam_pixel_dim'])

        pixel_t = np.linspace(0, (params['num_time_bins'] - 1) * params['bin_size'], params['num_time_bins'])

        X, Y = np.meshgrid(pixel_x, pixel_y)
        cam_pos = np.vstack([X.ravel(), Y.ravel(), np.zeros(params['cam_pixel_dim']**2)]).T

        X_ind, Y_ind = np.meshgrid(range(1, params['cam_pixel_dim'] + 1), range(1, params['cam_pixel_dim'] + 1))
        cam_pos_ind = np.vstack([X_ind.ravel(), Y_ind.ravel()]).T

        X, Y, T = np.meshgrid(pixel_x, pixel_y, pixel_t)
        cam_pixel = np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T

        # Form the corresponding measurement
        y_meas_vec = np.zeros(cam_pixel.shape[0])

        fourpi = 4 * np.pi * np.pi
        floor_normal = np.array([0, 0, 1])
        floor_pixel_width = params['camera_FOV'] / params['cam_pixel_dim']

        spix = scene_pixel.shape[0]
        for sp in range(spix):
            scene_center = scene_pixel[sp, :]
            m = (scene_center[1] - cam_pos[:, 1]) / (scene_center[0] - cam_pos[:, 0])
            b = scene_center[1] - m * scene_center[0]
            xint = -b / m

            noc = xint > 0
            if np.sum(noc) > 0:
                lps = params['laser_pos'] - scene_center
                fovsp = cam_pos[noc, :] - scene_center

                d1s = np.sum(lps**2, axis=1)
                d2s = np.sum(fovsp**2, axis=1)
                d1 = np.sqrt(d1s)
                d2 = np.sqrt(d2s)
                distance = d1 + d2
                tbin = distance / (params['c'] * params['bin_size'])
                arrival_bin = np.ceil(tbin).astype(int)

                dot1 = np.maximum(0, np.sum(scene_pixel_normal[sp, :] * lps / d1[:, np.newaxis], axis=1))
                dot2 = np.maximum(0, np.sum(scene_pixel_normal[sp, :] * fovsp / d2[:, np.newaxis], axis=1))
                dot3 = np.maximum(0, np.sum(params['laser_normal'] * -lps / d1[:, np.newaxis], axis=1))
                dot4 = np.maximum(0, np.sum(floor_normal * -fovsp / d2[:, np.newaxis], axis=1))

                intensity = laser_intensity * scene_pixel_area * (dot1 * dot2 * dot3 * dot4) / (fourpi * d1s * d2s)

                coord = (arrival_bin - 1) * params['cam_pixel_dim']**2 + (cam_pos_ind[noc, 0] - 1) * params['cam_pixel_dim'] + cam_pos_ind[noc, 1]
                y_meas_vec[coord] += intensity

                if sp % 50000 == 0:
                    print(f"{100 * sp / spix:.2f}%, t={time.time()}")

        y_meas_vec = y_meas_vec.reshape((params['cam_pixel_dim'], params['cam_pixel_dim'], params['num_time_bins']))

        plt.figure()
        plt.plot(np.squeeze(y_meas_vec[0, 0, :]))
        plt.show()

        filename = f"Video_{int(params['bin_size'] * 1e12)}ps_ThinFacet_pos_{xx}.mat"
        savemat(filename, {'params': params, 'y_meas_vec': y_meas_vec, 'objects': objects})

# Call main_generate_data with the appropriate coordinates
main_generate_data([-1, 1], [1.25, 1.25])
