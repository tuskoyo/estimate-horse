import os
from itertools import combinations

import pandas as pd
import plotly.graph_objects as go
from gpat.utils.files import FileName
from gpat.utils.skeleton_keypoints import keypoints_connections


def get_3d_motion_data(
    df: pd.DataFrame,
    frame: int,
) -> dict:
    target_dict = {}
    for part, (start, end) in keypoints_connections.items():
        for axis in ['x', 'y', 'z']:
            key = f"{part}_{axis}"
            target_dict[key] = df[[f"{start}_{axis}", f"{end}_{axis}"]].loc[frame]
    return target_dict

def plot_3d_motion(
    threed_data_path: str,
    line_width: int = 10,
    marker_size: int = 3,
    graph_mode: str = "lines+markers",
    frame_step: int = 1,
    point_show: bool = False,
) -> None:
    # Read the 3D motion data
    data_dir = os.path.expanduser(os.path.dirname(threed_data_path))
    output_path = os.path.join(data_dir, FileName.threed_motion)
    
    df = pd.read_csv(threed_data_path)
    df.ffill(inplace=True)
    df = df.loc[(df.filter(like="_x").sum(axis=1) != 0)]

    if point_show:
        point_columns = [col.replace("_x", "") for col in df.columns if col.endswith("_x") and col.startswith("POINT")]
        point_combinations = list(combinations(point_columns, 2))
        for i, (start, end) in enumerate(point_combinations):
            keypoints_connections[f"LINE{i+1}"] = [start, end]
    
    # Create the 3D motion plot
    x_max = df.filter(like="_x").max().max()
    x_min = df.filter(like="_x").min().min()
    y_max = df.filter(like="_y").max().max()
    y_min = df.filter(like="_y").min().min()
    z_max = df.filter(like="_z").max().max()
    z_min = df.filter(like="_z").min().min()
    min_frame = df.index.min()
    max_frame = df.index.max()
    
    frames = []
    for frame in range(min_frame, max_frame + 1, frame_step):
        print(f"\rProcessing frame: {frame}/{max_frame}", end="")
        vec_data = get_3d_motion_data(df, frame)

        x_vec_label = list(vec_data.keys())[0::3]
        y_vec_label = list(vec_data.keys())[1::3]
        z_vec_label = list(vec_data.keys())[2::3]
        vec_name = [label.replace("_x", "") for label in x_vec_label]

        fig = go.Frame(
            data=[
                go.Scatter3d(
                    x=vec_data[x_label],
                    y=vec_data[y_label],
                    z=vec_data[z_label],
                    mode=graph_mode,
                    line=dict(width=line_width),
                    marker=dict(size=marker_size),
                    name=name,
                )
                for x_label, y_label, z_label, name in zip(
                    x_vec_label, y_vec_label, z_vec_label, vec_name
                )
            ],
            name=f"{frame}",
            layout=go.Layout(title=f"frame:{frame}"),
        )
        frames.append(fig)
    print()

    vec_data = get_3d_motion_data(df, min_frame)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vec_data[x_label],
                y=vec_data[y_label],
                z=vec_data[z_label],
                mode=graph_mode,
                line=dict(width=line_width),
                marker=dict(size=marker_size),
                name=name,
            )
            for x_label, y_label, z_label, name in zip(
                x_vec_label, y_vec_label, z_vec_label, vec_name
            )
        ],
        frames=frames,
    )

    steps = []
    for frame in range(min_frame, max_frame + 1, frame_step):
        step = dict(
            method="animate",
            args=[
                [f"{frame}"],
                dict(frame=dict(duration=1, redraw=True), mode="immediate"),
            ],
            label=f"{frame}",
        )
        steps.append(step)

    sliders = [
        dict(
            steps=steps,
            active=0,
            transition=dict(duration=0),
            currentvalue=dict(
                font=dict(size=20), prefix="", visible=True, xanchor="right"
            ),
        )
    ]

    fig.update_layout(
        scene=dict(
            camera=dict(eye=dict(x=x_max, y=y_max, z=z_max)),
            xaxis=dict(title='X', range=[x_min, x_max]),
            yaxis=dict(title='Y', range=[y_min, y_max]),
            zaxis=dict(title='Z', range=[z_min, z_max]),
            aspectmode='manual',
            aspectratio=dict(
                x=(x_max - x_min),
                y=(y_max - y_min),
                z=(z_max - z_min)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                xanchor="left",
                yanchor="top",
                x=0,
                y=1,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=sliders,
    )
    
    fig.write_html(output_path, auto_play=False)

if __name__ == "__main__":
    threed_data_path = "/mnt/d/sasaki_20240930/data/center_171/3d_position_data.csv"
    plot_3d_motion(threed_data_path)