import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nmz(x):
    """Normalize vector"""
    return x / np.linalg.norm(x)

def plot_scene_dual_training_selfOcc(laser_pos, objects, params):
    showOcclusion = False  # if this is True, we don't plot the occluded scenery

    # Find camera pixel locations
    pixel_x = np.linspace(
        params['camera_FOV_center'][0] - params['camera_FOV'] / 2 + params['camera_FOV'] / (2 * params['cam_pixel_dim']),
        params['camera_FOV_center'][0] + params['camera_FOV'] / 2 - params['camera_FOV'] / (2 * params['cam_pixel_dim']),
        params['cam_pixel_dim']
    )
    pixel_y = np.linspace(
        params['camera_FOV_center'][1] - params['camera_FOV'] / 2 + params['camera_FOV'] / (2 * params['cam_pixel_dim']),
        params['camera_FOV_center'][1] + params['camera_FOV'] / 2 - params['camera_FOV'] / (2 * params['cam_pixel_dim']),
        params['cam_pixel_dim']
    )
    X, Y = np.meshgrid(pixel_x, pixel_y)
    cam_pixel = np.vstack([X.ravel(), Y.ravel(), np.zeros(params['cam_pixel_dim']**2)]).T

    # Discretize
    numpatches = 0
    for o in range(len(objects)):
        if objects[o][0] == 'wall':
            numpatches += round(np.linalg.norm(objects[o][2]) / params['wall_discr']) * round(np.linalg.norm(objects[o][3]) / params['wall_discr'])
        else:
            numpatches += objects[o][3] * 0.5 * np.pi * objects[o][2] / params['cyl_discr']

    scene_pixel = np.zeros((int(np.ceil(numpatches)), 3))
    scene_pixel_normal = np.zeros((int(np.ceil(numpatches)), 3))
    scene_pixel_area = (params['wall_discr'])**2
    scene_pixel_angle = 1000 * np.ones(int(np.ceil(numpatches)))

    v = nmz(laser_pos[:2])
    light_angle = v[0]
    patch_count = 0

    for o in range(len(objects)):
        if objects[o][0] == 'wall':
            for i in np.linspace(0, 1, round(np.linalg.norm(objects[o][2]) / params['wall_discr'])):
                for j in np.linspace(0, 1, round(np.linalg.norm(objects[o][3]) / params['wall_discr'])):
                    if objects[o][1][2] + i * objects[o][2][2] + j * objects[o][3][2] > 0:
                        pos = objects[o][1] + i * objects[o][2] + j * objects[o][3]

                        oc = False

                        if o <= 4:
                            for k in range(5, len(objects)):
                                I, check = plane_line_intersect(objects[k][4], objects[k][1], laser_pos, pos)

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

                        if showOcclusion:
                            if -scene_pixel_angle + light_angle < 0:
                                oc = True

                        if not oc:
                            scene_pixel[patch_count] = pos
                            scene_pixel_normal[patch_count] = objects[o][4]
                            patch_count += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axes.set_xlim3d(left=-2, right=2) 
    ax.axes.set_ylim3d(bottom=-1, top=3) 
    ax.axes.set_zlim3d(bottom=0, top=3) 
    ax.scatter(scene_pixel[:, 0], scene_pixel[:, 1], scene_pixel[:, 2], c='b', marker='.')
    ax.scatter(laser_pos[0], laser_pos[1], laser_pos[2], c='r', marker='o')
    ax.scatter(cam_pixel[:, 0], cam_pixel[:, 1], cam_pixel[:, 2], c='g', marker='*')

    ax.view_init(elev=90, azim=0)
    ax.set_aspect('auto')
    plt.grid(True)
    plt.show()

