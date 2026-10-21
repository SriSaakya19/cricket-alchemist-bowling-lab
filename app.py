        else:
            neutral_pairs.append((a, b))

    plot_points = [
        None if pose is None
        else np.column_stack([pose[:, 0], pose[:, 2], -pose[:, 1]])
        for pose in display_poses
    ]
    valid = [pts for pts in plot_points if pts is not None]
    stacked = np.stack(valid)
    half_x = float(np.clip(np.abs(stacked[:, :, 0]).max() * 1.1, 0.5, 1.6))
    half_y = float(np.clip(np.abs(stacked[:, :, 1]).max() * 1.1, 0.5, 1.6))
    z_low = float(stacked[:, :, 2].min() - 0.08)
    z_high = float(stacked[:, :, 2].max() + 0.08)

    time_low, time_high = float(min(times)), float(max(times))
    if time_high - time_low < 1e-6:
        time_high = time_low + 1.0

    empty = ([], [], [])
    empty_mesh = ([0.0] * 4, [0.0] * 4, [0.0] * 4)

    def segments(points, pairs, flat_z=None):
        xs, ys, zs = [], [], []
        for a, b in pairs:
            za = points[a][2] if flat_z is None else flat_z
            zb = points[b][2] if flat_z is None else flat_z
            xs += [round(float(points[a][0]), 3), round(float(points[b][0]), 3), None]
            ys += [round(float(points[a][1]), 3), round(float(points[b][1]), 3), None]
            zs += [round(float(za), 3), round(float(zb), 3), None]
        return xs, ys, zs

    def frame_traces(index):
        points = plot_points[index]
        if points is None:
            return [empty, empty, empty, empty, empty, empty_mesh, empty, empty]

        side = sides[index]
        torso = points[[11, 12, 24, 23]]
        joints = points[joint_ids]

        trail = [], [], []
        wrist_id = {"right": 16, "left": 15}.get(side)
        if wrist_id is not None:
            for j in range(max(0, index - trail_length + 1), index + 1):
                if plot_points[j] is not None:
                    trail[0].append(round(float(plot_points[j][wrist_id][0]), 3))
                    trail[1].append(round(float(plot_points[j][wrist_id][1]), 3))
                    trail[2].append(round(float(plot_points[j][wrist_id][2]), 3))

        return [
            segments(points, neutral_pairs),
            segments(points, left_pairs),
            segments(points, right_pairs),
            segments(points, arm_pairs.get(side, [])),
            ([round(float(v), 3) for v in joints[:, 0]],
             [round(float(v), 3) for v in joints[:, 1]],
             [round(float(v), 3) for v in joints[:, 2]]),
            ([round(float(v), 3) for v in torso[:, 0]],
             [round(float(v), 3) for v in torso[:, 1]],
             [round(float(v), 3) for v in torso[:, 2]]),
            segments(points, connections, flat_z=z_low),
            trail,
        ]

    def cursor_traces(index):
        moment = float(times[index])
        value = elbow_3d[index]
        marker = ([moment], [value]) if value is not None else ([], [])
        return [marker, ([moment, moment], [graph_low, graph_high])]

    def readout_annotation(index):
        return [dict(
            text=readouts[index], xref="paper", yref="paper",
            x=1.0, y=1.0, xanchor="right", yanchor="top", showarrow=False,
            align="right", font=dict(color="#FFD16D", size=13),
            bgcolor="rgba(8,20,36,0.75)", bordercolor="#294968",
            borderwidth=1, borderpad=8,
        )]

    first = next(i for i in range(frame_count) if plot_points[i] is not None)
    t0 = frame_traces(first)
    c0 = cursor_traces(first)

    figure = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "scene"}], [{"type": "xy"}]],
        row_heights=[0.8, 0.2],
        vertical_spacing=0.04,
    )

    first_row_traces = [
        go.Scatter3d(x=t0[0][0], y=t0[0][1], z=t0[0][2], mode="lines",
                     line=dict(color="#B8CCE0", width=6), hoverinfo="skip",
                     showlegend=False),
        go.Scatter3d(x=t0[1][0], y=t0[1][1], z=t0[1][2], mode="lines",
                     line=dict(color="#4CC1FF", width=8), hoverinfo="skip",
                     showlegend=False),
        go.Scatter3d(x=t0[2][0], y=t0[2][1], z=t0[2][2], mode="lines",
                     line=dict(color="#FF7F96", width=8), hoverinfo="skip",
                     showlegend=False),
        go.Scatter3d(x=t0[3][0], y=t0[3][1], z=t0[3][2], mode="lines",
                     line=dict(color="#FFD16D", width=13), hoverinfo="skip",
                     showlegend=False),
        go.Scatter3d(x=t0[4][0], y=t0[4][1], z=t0[4][2], mode="markers",
                     marker=dict(size=[14] + [5] * (len(joint_ids) - 1),
                                 color="#FFFFFF",
                                 line=dict(color="#0A1828", width=1)),
                     hoverinfo="skip", showlegend=False),
        go.Mesh3d(x=t0[5][0], y=t0[5][1], z=t0[5][2],
                  i=[0, 0], j=[1, 2], k=[2, 3],
                  color="#7FB8E6", opacity=0.35, flatshading=True,
                  hoverinfo="skip", showlegend=False),
        go.Scatter3d(x=t0[6][0], y=t0[6][1], z=t0[6][2], mode="lines",
                     line=dict(color="rgba(130,170,210,0.28)", width=6),
                     hoverinfo="skip", showlegend=False),
        go.Scatter3d(x=t0[7][0], y=t0[7][1], z=t0[7][2], mode="lines",
                     line=dict(color="rgba(255,209,109,0.65)", width=4),
                     hoverinfo="skip", showlegend=False),
    ]
    for trace in first_row_traces:
        figure.add_trace(trace, row=1, col=1)

    # trace 8 and 9: the full elbow graph (static); 10 and 11: the moving cursor
    figure.add_trace(
        go.Scatter(x=list(times), y=list(elbow_3d), mode="lines",
                   name="Elbow (3D estimate)", connectgaps=False,
                   line=dict(color="#4CC1FF", width=3), hoverinfo="skip"),
        row=2, col=1)
    figure.add_trace(
        go.Scatter(x=list(times), y=list(elbow_2d), mode="lines",
                   name="Elbow (2D image)", connectgaps=False,
                   line=dict(color="rgba(184,204,224,0.6)", width=2, dash="dot"),
                   hoverinfo="skip"),
        row=2, col=1)
    figure.add_trace(
        go.Scatter(x=c0[0][0], y=c0[0][1], mode="markers",
                   marker=dict(size=11, color="#FFD16D",
                               line=dict(color="#FFFFFF", width=1)),
                   hoverinfo="skip", showlegend=False),
        row=2, col=1)
    figure.add_trace(
        go.Scatter(x=c0[1][0], y=c0[1][1], mode="lines",
                   line=dict(color="rgba(255,209,109,0.85)", width=2),
                   hoverinfo="skip", showlegend=False),
        row=2, col=1)

    animated_traces = list(range(8)) + [10, 11]
    figure.frames = [
        go.Frame(
            name=str(i),
            traces=animated_traces,
            data=[
                go.Mesh3d(x=part[0], y=part[1], z=part[2]) if n == 5
                else go.Scatter3d(x=part[0], y=part[1], z=part[2]) if n < 8
                else go.Scatter(x=part[0], y=part[1])
                for n, part in enumerate(frame_traces(i) + cursor_traces(i))
            ],
            layout=dict(annotations=readout_annotation(i)),
        )
