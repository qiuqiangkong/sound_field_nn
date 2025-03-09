import numpy as np
import h5py


def get_pressure(path):
    loaded = np.load(path,allow_pickle=True)
    loaded_dict = loaded.item()

    data = loaded_dict["p"]
    data = data.reshape(-1,128,128).transpose(0,2,1)
    return data


def add():

    h5_path = "/datasets/paws/temp_hdf5/2024-06-27-14-52-26_kwave_output.h5"

    x = get_pressure("/public/acoustic_field_data/training_set/complicated_room1_save1.npy")

    # with h5py.File(h5_path, 'r') as hf:
    #     print(hf.keys())
    #     from IPython import embed; embed(using=False); os._exit(0)

    # pass
    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    h5_path = "/datasets/paws/temp_hdf5/2024-06-27-14-52-26_kwave_output.h5"

    with h5py.File(h5_path, 'r') as hf:
        print(hf.keys())
        from IPython import embed; embed(using=False); os._exit(0)


def add3():

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Simulation Parameters
    nx, ny = 100, 100  # Grid size
    nt = 200  # Number of time steps
    c = 1.0  # Wave speed
    dx = dy = 1.0  # Spatial step
    dt = 0.5  # Time step
    r = (c * dt / dx)  # CFL condition (must be <= 1/sqrt(2))

    # Initialize wave field
    u = np.zeros((nx, ny, 3))  # u[i, j, n] where n=0,1,2 (current, previous, next)

    # Initial condition: Gaussian pulse at center
    x0, y0 = nx // 2, ny // 2
    sigma = 5.0
    for i in range(nx):
        for j in range(ny):
            u[i, j, 1] = np.exp(-((i - x0)**2 + (j - y0)**2) / (2 * sigma**2))

    # FDTD Iteration
    for n in range(1, nt - 1):  # Time loop
        u[1:-1, 1:-1, 2] = (2 * u[1:-1, 1:-1, 1] - u[1:-1, 1:-1, 0] +
                             r**2 * (u[2:, 1:-1, 1] + u[:-2, 1:-1, 1] +
                                     u[1:-1, 2:, 1] + u[1:-1, :-2, 1] -
                                     4 * u[1:-1, 1:-1, 1]))

        # Rotate time indices: old ← previous, current ← next
        u[:, :, 0] = u[:, :, 1]
        u[:, :, 1] = u[:, :, 2]

    # Save animation to video
    fps = 30  # Frames per second
    output_filename = "fdtd_simulation.mp4"

    writer = animation.FFMpegWriter(fps=fps)
    ani = animation.FuncAnimation(fig, update, frames=nt, interval=30, blit=True)

    print(f"Saving animation to {output_filename}...")
    ani.save(output_filename, writer=writer)
    print("Video saved successfully!")


if __name__ == "__main__":

    add3()