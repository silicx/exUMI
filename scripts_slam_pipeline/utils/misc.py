import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")


__all__ = [
    "get_single_path",
    "listOfDict_to_dictOfList",
    "custom_minimize",
    "plot_trajectories",
]


def get_single_path(path_glob):
    path_glob = list(path_glob)
    if len(path_glob) == 0:
        raise FileNotFoundError(f"No files/dirs found in {path_glob}")
    elif len(path_glob) > 1:
        raise OSError(f"Multiple files/dirs found in {path_glob}.")

    return path_glob[0]


def listOfDict_to_dictOfList(listOfDict):
    dictOfList = {}
    length = None
    for key in listOfDict[0].keys():
        dictOfList[key] = [x[key] for x in listOfDict]
        if length is None:
            length = len(dictOfList[key])
        else:
            assert length == len(dictOfList[key]), f"Data length mismatch: {length} vs {len(dictOfList[key])} in {key}"
    return dictOfList



def custom_minimize(mse_error, foo, bounds):
    class Foo:
        x = None

    left_offset_bound, right_offset_bound = bounds[0]
    while right_offset_bound - left_offset_bound > 0.0001:
        xs = np.linspace(left_offset_bound, right_offset_bound, 10)
        ys = [mse_error([x]) for x in xs]
        min_idx = np.argmin(ys)
        left_offset_bound = xs[max(0, min_idx-2)]
        right_offset_bound = xs[min(len(xs)-1, min_idx+2)]
        print("left", left_offset_bound, "right", right_offset_bound, "error", ys[min_idx])
    
    res = Foo()
    res.x = np.array([(left_offset_bound + right_offset_bound) / 2])
    return res




def plot_trajectories(aruco_trajectory, arcap_trajectory, turn_point_aruco, turn_point_arcap, save_path):
    plt.figure(figsize=(4, 3))
    plt.plot([d[0] for d in aruco_trajectory], [d[1] for d in aruco_trajectory], label='ArUco', color="#6A80B9")
    plt.plot([d[0] for d in arcap_trajectory], [d[1] for d in arcap_trajectory], label='AR-MoCap', color="#F6C794")

    # draw the turning points as scatter points:
    tms = [(t, v) for t, d, v in turn_point_aruco if d > 0] 
    plt.scatter([t for t,v in tms], [v for t,v in tms], c='r', s=50)
    tms = [(t, v) for t, d, v in turn_point_aruco if d < 0] 
    plt.scatter([t for t,v in tms], [v for t,v in tms], c='b', s=50)

    tms = [(t, v) for t, d, v in turn_point_arcap if d > 0] 
    plt.scatter([t for t,v in tms], [v for t,v in tms], c='r', s=50)
    tms = [(t, v) for t, d, v in turn_point_arcap if d < 0] 
    plt.scatter([t for t,v in tms], [v for t,v in tms], c='b', s=50)
    
    # print([(t, d) for t,d,v in turn_point_aruco])
    # print([(t, d) for t,d,v in turn_point_arcap])
    
    vals = [d[1] for d in aruco_trajectory] + [d[1] for d in arcap_trajectory]
    mx, mn = np.max(vals), np.min(vals)
    dt = (mx - mn) * 0.1
    plt.ylim(mn-dt, mx+dt)

    # plt.xticks([])
    # plt.yticks([])

    plt.xlabel('Time')
    plt.ylabel('X-axis Position')
    
    plt.legend()
    plt.savefig(str(save_path))
    plt.close()




def plot_long_horizon_trajectory(gopro_trajectory, another_trajectory, title, save_dir):
    fig, axs = plt.subplots(1)
    fig.suptitle(title)
    axs.plot([t for t,_ in gopro_trajectory], [v for _,v in gopro_trajectory], label='GoPro')
    # axs.set_xlim(aruco_trajectory[0][0]-extend_range, aruco_trajectory[-1][0]+extend_range)

    last_val = []
    for i,(t,v) in enumerate(another_trajectory):
        plot_val = (i == len(another_trajectory)-1)

        if len(last_val) == 0:
            last_val.append((t,v))
        elif (v is None) == (last_val[-1][1] is None):
            last_val.append((t,v))
        else:
            plot_val = True
        
        if plot_val:
            color = "gray" if last_val[-1][1] is None else "orange"
            axs.plot([t for t,_ in last_val], [(0 if v is None else v) for _,v in last_val], color=color)
            last_val = []
    
    plt.legend()
    fig.savefig(str(save_dir))
    plt.close()