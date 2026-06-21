import imageio
import kmapper

# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def graph_km(
    data, path, projection=PCA(3), title="Title", nr_cubes=15, overlap_perc=0.5, clusterer=DBSCAN(1)
):
    mapper = kmapper.KeplerMapper(verbose=1)
    projected_data = mapper.fit_transform(data, projection=projection, scaler=None)
    cover = kmapper.Cover(n_cubes=nr_cubes, perc_overlap=overlap_perc)
    graph = mapper.map(projected_data, data, cover=cover, clusterer=clusterer)
    html = mapper.visualize(graph, path_html=path, title=title)  # noqa: F841
    # color = label[:,1]
    # html = mapper.visualize(graph,
    #                        path_html=path,
    #                        title=title)


def multiple_model_pca_plot(param_history_list, archs, save_name="no_name"):
    """
    Assume models have same layer number
    """
    training_weights_list = []
    for param_history in param_history_list:
        training_weights = []  # from model, timestep, layer to model, layer, timestep
        for param in param_history:
            for l, layer in enumerate(param):  # noqa: E741
                layer_weights = np.array(layer)
                if l >= len(training_weights):
                    training_weights.append([layer_weights])
                else:
                    training_weights[l].append(layer_weights)
        training_weights_list.append(training_weights)

    for lay_num in range(len(training_weights_list[0])):  # assume all models have same layers
        curr_layer_weight_history_all_models = []
        neuron_number_all_models = []
        training_step_number_all_models = []
        all_model_num_vecs = []

        for model_num, training_weights in enumerate(training_weights_list):
            curr_layer_weight_history = np.concatenate(training_weights[lay_num])
            curr_layer_weight_history_all_models.append(curr_layer_weight_history)
            num_neurons = archs[model_num][lay_num + 1]
            num_layer_weights_all_epochs = curr_layer_weight_history.shape[0]
            num_saved_weights = num_layer_weights_all_epochs / num_neurons  # noqa: F841
            neuron_number = np.arange(num_layer_weights_all_epochs) % num_neurons
            training_step_number = np.arange(num_layer_weights_all_epochs) / num_neurons
            neuron_number_all_models.append(neuron_number)
            training_step_number_all_models.append(training_step_number)
            model_num_vec = np.ones(num_layer_weights_all_epochs) * model_num
            all_model_num_vecs.append(model_num_vec)

        curr_layer_weight_history_all_models = np.concatenate(curr_layer_weight_history_all_models)
        neuron_number_all_models = np.concatenate(neuron_number_all_models)
        training_step_number_all_models = np.concatenate(training_step_number_all_models)
        all_model_num_vecs = np.concatenate(all_model_num_vecs)

        X_pca2 = PCA(n_components=2).fit_transform(curr_layer_weight_history_all_models)
        plt.figure(figsize=(8, 4))
        plt.subplot(131)
        plt.scatter(
            X_pca2[:, 0],
            X_pca2[:, 1],
            s=3,
            c=training_step_number_all_models,
            cmap=plt.cm.get_cmap("viridis"),
            alpha=0.8,
        )
        plt.title("Layer {} (colored by training step)".format(lay_num))
        plt.subplot(132)
        plt.scatter(
            X_pca2[:, 0],
            X_pca2[:, 1],
            s=4,
            c=neuron_number_all_models,
            cmap=plt.cm.get_cmap("jet"),
            alpha=0.8,
        )
        plt.title("Layer {} (colored by neuron)".format(lay_num))
        plt.subplot(133)
        plt.scatter(
            X_pca2[:, 0],
            X_pca2[:, 1],
            s=4,
            c=all_model_num_vecs,
            cmap=plt.cm.get_cmap("jet"),
            alpha=0.8,
        )
        plt.title("Layer {} (colored by model number)".format(lay_num))
        plt.tight_layout()
        plt.savefig(save_name + "_layer_" + str(lay_num) + "_pca.png")
        plt.close()
        # plt.show()


def mapper_plot(param_history, arch, save_name="no_name"):
    training_weights = []  # timestep, layer to layer, timestep
    for param in param_history:
        for l, layer in enumerate(param):  # noqa: E741
            layer_weights = np.array(layer)
            if l >= len(training_weights):
                training_weights.append([layer_weights])
            else:
                training_weights[l].append(layer_weights)

    for lay_num in range(len(training_weights)):
        curr_layer_weight_history = np.concatenate(training_weights[lay_num])
        X_pca2 = PCA(n_components=2).fit_transform(curr_layer_weight_history)
        num_neurons = arch[lay_num + 1]
        num_saved_weights = X_pca2.shape[0] / num_neurons  # noqa: F841
        neuron_number = np.arange(X_pca2.shape[0]) % num_neurons
        training_step_number = np.arange(X_pca2.shape[0]) / num_neurons
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.scatter(
            X_pca2[:, 0],
            X_pca2[:, 1],
            s=3,
            c=training_step_number,
            cmap=plt.cm.get_cmap("viridis"),
            alpha=0.8,
        )
        plt.title("Layer {} (colored by training step)".format(lay_num))
        plt.subplot(122)
        plt.scatter(
            X_pca2[:, 0], X_pca2[:, 1], s=4, c=neuron_number, cmap=plt.cm.get_cmap("jet"), alpha=0.8
        )
        plt.title("Layer {} (colored by neuron)".format(lay_num))
        plt.tight_layout()
        plt.savefig(save_name + "_layer_" + str(lay_num) + "_pca.png")
        plt.close()
        # plt.show()

    """
    for l, layer in enumerate(training_weights):
        curr_model_weight_history = np.concatenate(training_weights[l])
        # Initialize
        # dimensionality reduction
        #pca = PCA(curr_model_weight_history.shape[1])
        pca = PCA(n_components=20)
        pca_feats = pca.fit_transform(curr_model_weight_history)
        clusterer = DBSCAN(eps=0.3, min_samples=2)
        overlap_perc = 0.97
        nr_cubes = 15
        graph_km(pca_feats,
                 projection='l2norm',
                 path=save_name + "_layer_" + str(l) + "_weights_mapper_output.html",
                 title=save_name + '_layer_' + str(l),
                 nr_cubes=nr_cubes,
                 overlap_perc=overlap_perc)

        # Fit to and transform the data
        #projected_data = mapper.fit_transform(pca_feats, projection=[0,1]) # X-Y axis
        #projected_data = mapper.fit_transform(curr_model_weight_history, projection=[0,1]) # X-Y axis
        #projected_data = mapper.fit_transform(curr_model_weight_history, projection='l2norm') # X-Y axis
        projected_data = mapper.fit_transform(pca_feats, projection='l2norm', scaler=None) # X-Y axis

        # Create a cover with 10 elements
        #cover = kmapper.Cover(n_cubes=10)
        cover = kmapper.Cover(n_cubes=30, perc_overlap=0.99)

        # Create dictionary called 'graph' with nodes, edges and meta-information
        graph = mapper.map(projected_data, curr_model_weight_history, cover=cover, clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=2))

        # Visualize it
        mapper.visualize(graph, path_html=save_name + "_layer_" + str(l) + "_weights_mapper_output.html", title=save_name + '_layer_' + str(l))
        """


import matplotlib.animation as animation  # noqa: E402


def animate_network_weights_and_hidden_states_optimized(
    weights_list,
    weights_encodings_list,
    neurons_per_layer,
    save_name="weight_states.gif",
    verbose=True,
):
    if verbose:
        print("Making weight state gif and saving to " + str(save_name))

    # Set up the colormap and normalization
    cmap = plt.get_cmap("viridis")
    all_hidden_states = np.concatenate(
        [enc.flatten() for weights_encodings in weights_encodings_list for enc in weights_encodings]
    )
    all_weights = np.concatenate(
        [weight.flatten() for weights in weights_list for weight in weights]
    )
    all_data = np.concatenate([all_hidden_states, all_weights])
    norm = Normalize(vmin=all_data.min(), vmax=all_data.max())
    scalar_mappable = ScalarMappable(cmap=cmap, norm=norm)

    # Precompute node positions
    node_positions = {}
    for layer_idx, num_neurons in enumerate(neurons_per_layer):
        x_position = layer_idx * 0.2  # Horizontal gap
        y_positions = np.linspace(0, 1, num_neurons + 2)[1:-1]
        for neuron_idx in range(num_neurons):
            node_positions[(layer_idx, neuron_idx)] = (x_position, y_positions[neuron_idx])

    # Initialize the figure and axes
    encoding_dim = weights_encodings_list[0][0].shape[-1] + 1  # plus one for weight itself
    fig, axs = plt.subplots(4, encoding_dim // 4 + 1, figsize=(15, 10))
    fig.subplots_adjust(right=0.8)  # Adjusting right to make space for the colorbar
    colorbar_ax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    fig.colorbar(scalar_mappable, cax=colorbar_ax)

    # Create lists to hold line objects
    lines = []
    for subplot_idx in range(encoding_dim):
        ax = axs[subplot_idx // (encoding_dim // 4 + 1), subplot_idx % (encoding_dim // 4 + 1)]
        line_objs = []
        for layer_idx in range(1, len(neurons_per_layer)):
            for prev_neuron in range(neurons_per_layer[layer_idx - 1]):
                for next_neuron in range(neurons_per_layer[layer_idx]):
                    (line,) = ax.plot([], [], color="blue", linewidth=2)
                    line_objs.append(line)
        lines.append(line_objs)

    scatter_objs = [ax.scatter(*zip(*node_positions.values()), color="k") for ax in axs.flat]

    def update(frame):
        weights = weights_list[frame]
        weights_encodings = weights_encodings_list[frame]
        for subplot_idx, line_objs in enumerate(lines):
            for line, (layer_idx, prev_neuron, next_neuron) in zip(
                line_objs,
                [
                    (layer_idx, prev_neuron, next_neuron)
                    for layer_idx in range(1, len(neurons_per_layer))
                    for prev_neuron in range(neurons_per_layer[layer_idx - 1])
                    for next_neuron in range(neurons_per_layer[layer_idx])
                ],
            ):
                src = node_positions[(layer_idx - 1, prev_neuron)]
                dst = node_positions[(layer_idx, next_neuron)]
                if subplot_idx == encoding_dim - 1:
                    weight = weights[layer_idx - 1][next_neuron, prev_neuron]
                else:
                    weight = weights_encodings[layer_idx - 1][next_neuron, prev_neuron, subplot_idx]
                color = cmap(norm(weight))
                line.set_data([src[0], dst[0]], [src[1], dst[1]])
                line.set_color(color)
        return [item for sublist in lines for item in sublist] + scatter_objs

    ani = animation.FuncAnimation(fig, update, frames=len(weights_list), blit=True)
    ani.save(save_name, writer="imagemagick", fps=2)
    plt.close(fig)


"""
def animate_network_weights_and_hidden_states_interactive(weights_list, weights_encodings_list, neurons_per_layer, save_name='interactive_weight_states.gif', verbose=True):
    if verbose:
        print('Creating interactive weight state animation.')

    # Set up the colormap and normalization
    cmap = plt.get_cmap('viridis')
    all_hidden_states = np.concatenate([enc.flatten() for weights_encodings in weights_encodings_list for enc in weights_encodings])
    all_weights = np.concatenate([weight.flatten() for weights in weights_list for weight in weights])
    all_data = np.concatenate([all_hidden_states, all_weights])
    norm = Normalize(vmin=all_data.min(), vmax=all_data.max())
    scalar_mappable = ScalarMappable(cmap=cmap, norm=norm)

    # Precompute node positions
    node_positions = {}
    for layer_idx, num_neurons in enumerate(neurons_per_layer):
        x_position = layer_idx * 0.2  # Horizontal gap
        y_positions = np.linspace(0, 1, num_neurons + 2)[1:-1]
        for neuron_idx in range(num_neurons):
            node_positions[(layer_idx, neuron_idx)] = (x_position, y_positions[neuron_idx])

    # Initialize the figure and axes
    encoding_dim = weights_encodings_list[0][0].shape[-1] + 1 # plus one for weight itself
    fig, axs = plt.subplots(4, encoding_dim // 4 + 1, figsize=(15, 10))
    fig.subplots_adjust(right=0.8)  # Adjusting right to make space for the colorbar
    colorbar_ax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    fig.colorbar(scalar_mappable, cax=colorbar_ax)

    # Create lists to hold line objects and their data
    lines = []
    line_data = []  # To store the corresponding weight indices
    for subplot_idx in range(encoding_dim):
        ax = axs[subplot_idx // (encoding_dim // 4 + 1), subplot_idx % (encoding_dim // 4 + 1)]
        line_objs = []
        data_objs = []
        for layer_idx in range(1, len(neurons_per_layer)):
            for prev_neuron in range(neurons_per_layer[layer_idx-1]):
                for next_neuron in range(neurons_per_layer[layer_idx]):
                    line, = ax.plot([], [], color='blue', linewidth=2)
                    line_objs.append(line)
                    data_objs.append((layer_idx-1, next_neuron, prev_neuron, subplot_idx))
        lines.append(line_objs)
        line_data.append(data_objs)

    scatter_objs = [ax.scatter(*zip(*node_positions.values()), color='k') for ax in axs.flat]

    # Create a mutable container to store the current frame index and control flags
    current_frame = [0]
    reset_schedule = []
    dragging = [False]
    paused = [False]

    def update(frame):
        if paused[0]:
            return
        current_frame[0] = frame  # Update the current frame index

        # Apply scheduled resets
        if reset_schedule:
            for layer_idx, next_neuron, prev_neuron in reset_schedule:
                weights_list[frame][layer_idx] = weights_list[frame][layer_idx].at[next_neuron, prev_neuron].set(0)
            reset_schedule.clear()

        weights = weights_list[frame]
        weights_encodings = weights_encodings_list[frame]
        for subplot_idx, (line_objs, data_objs) in enumerate(zip(lines, line_data)):
            for line, (layer_idx, next_neuron, prev_neuron, enc_idx) in zip(line_objs, data_objs):
                src = node_positions[(layer_idx, prev_neuron)]
                dst = node_positions[(layer_idx + 1, next_neuron)]
                if enc_idx == encoding_dim - 1:
                    weight = weights[layer_idx][next_neuron, prev_neuron]
                else:
                    weight = weights_encodings[layer_idx][next_neuron, prev_neuron, enc_idx]
                if weight == 0:
                    color = 'red'
                else:
                    color = cmap(norm(weight))
                line.set_data([src[0], dst[0]], [src[1], dst[1]])
                line.set_color(color)
        return [item for sublist in lines for item in sublist] + scatter_objs

    def on_click(event):
        if event.button == 1:  # Left mouse button
            dragging[0] = True
            paused[0] = True  # Pause the animation

    def on_release(event):
        if event.button == 1:  # Left mouse button
            dragging[0] = False
            paused[0] = False  # Resume the animation

    def on_motion(event):
        if dragging[0]:
            frame = current_frame[0]  # Get the current frame index
            for subplot_idx, (line_objs, data_objs) in enumerate(zip(lines, line_data)):
                for line, (layer_idx, next_neuron, prev_neuron, enc_idx) in zip(line_objs, data_objs):
                    if enc_idx == encoding_dim - 1:
                        x_data, y_data = line.get_data()
                        if len(x_data) > 0 and len(y_data) > 0:
                            x_click, y_click = event.xdata, event.ydata
                            dist = np.sqrt((x_click - np.mean(x_data))**2 + (y_click - np.mean(y_data))**2)
                            if dist < 0.05:  # Threshold distance to consider a click on the line
                                print(f'Scheduling reset for weight at layer {layer_idx}, from neuron {prev_neuron} to neuron {next_neuron} for next frame')
                                reset_schedule.append((layer_idx, next_neuron, prev_neuron))

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    ani = animation.FuncAnimation(fig, update, frames=len(weights_list), blit=True)
    plt.show()

"""


def animate_network_weights_and_hidden_states(
    weights_list,
    weights_encodings_list,
    neurons_per_layer,
    save_name="weight_states.gif",
    verbose=True,
):
    print("Making weight state gif and saving to " + str(save_name))
    # Set up the colormap
    cmap = plt.get_cmap("viridis")

    # Prepare to capture frames for the GIF
    frames = []
    all_hidden_states = np.concatenate(
        [enc.flatten() for weights_encodings in weights_encodings_list for enc in weights_encodings]
    )
    all_weights = np.concatenate(
        [weight.flatten() for weights in weights_list for weight in weights]
    )
    all_data = np.concatenate([all_hidden_states, all_weights])
    global_min = np.min(all_data)  # noqa: F841
    global_max = np.max(all_data)  # noqa: F841
    norm = Normalize(vmin=all_data.min(), vmax=all_data.max())
    scalar_mappable = ScalarMappable(cmap=cmap, norm=norm)

    # Process each set of weights and encodings
    for weights, weights_encodings in zip(weights_list, weights_encodings_list):
        encoding_dim = weights_encodings[0].shape[-1] + 1  # plus one for weight itself
        fig, axs = plt.subplots(4, encoding_dim // 4 + 1, figsize=(15, 10))
        fig.subplots_adjust(right=0.8)  # Adjusting right to make space for the colorbar

        # Calculate positions for neurons
        node_positions = {}
        for layer_idx, num_neurons in enumerate(neurons_per_layer):
            x_position = layer_idx * 0.2  # Horizontal gap
            y_positions = np.linspace(0, 1, num_neurons + 2)[1:-1]
            for neuron_idx in range(num_neurons):
                node_positions[(layer_idx, neuron_idx)] = (x_position, y_positions[neuron_idx])

        # Plot each weight and its encoding
        for subplot_idx in range(encoding_dim):
            ax = axs[subplot_idx // 2, subplot_idx % 2]
            for layer_idx in range(1, len(neurons_per_layer)):
                for prev_neuron in range(neurons_per_layer[layer_idx - 1]):
                    for next_neuron in range(neurons_per_layer[layer_idx]):
                        src = node_positions[(layer_idx - 1, prev_neuron)]
                        dst = node_positions[(layer_idx, next_neuron)]
                        if subplot_idx == encoding_dim - 1:
                            weight = weights[layer_idx - 1][next_neuron, prev_neuron]
                        else:
                            weight = weights_encodings[layer_idx - 1][
                                next_neuron, prev_neuron, subplot_idx
                            ]
                        color = cmap(norm(weight))  # Map this encoding dimension's value to color
                        ax.plot([src[0], dst[0]], [src[1], dst[1]], color=color, linewidth=2)

            for pos in node_positions.values():
                ax.scatter(*pos, color="k")

        colorbar_ax = axs[-1, -1]
        fig.colorbar(scalar_mappable, cax=colorbar_ax, orientation="horizontal")
        # Save the plot to a buffer (PNG) and then close the plot
        plt.savefig("temp.png")
        plt.close(fig)
        frames.append(imageio.imread("temp.png"))

    # Create GIF
    # imageio.mimsave(save_name, frames, fps=2, loop=0)
    imageio.mimsave(save_name, frames, duration=500, loop=0)


def plot_network_weights_and_hidden_states(
    weights, weights_encodings, neurons_per_layer, save_name="weight_states.png"
):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")  # Colormap

    # Position nodes and plot weights
    node_positions = {}  # To store positions of neurons for plotting
    y_gap = 0.1  # Vertical gap between layers  # noqa: F841
    x_gap = 0.2  # Horizontal gap within layers

    # Flatten weights_encodings for global norm calculation
    all_encodings_flat = np.vstack([enc.reshape(-1, enc.shape[-1]) for enc in weights_encodings])
    weights_flat = np.concatenate([weight_mat.flatten() for weight_mat in weights])
    all_weight_numbers = np.concatenate([all_encodings_flat.ravel(), weights_flat])
    global_norm = np.linalg.norm(all_weight_numbers)  # noqa: F841

    norm = Normalize(
        vmin=all_weight_numbers.min(), vmax=all_weight_numbers.max()
    )  # get norm to create scalarMappable
    scalar_mappable = ScalarMappable(cmap=cmap, norm=norm)

    # Calculate positions for neurons
    for layer_idx, num_neurons in enumerate(neurons_per_layer):
        x_position = layer_idx * x_gap
        y_positions = np.linspace(0, 1, num_neurons + 2)[1:-1]
        for neuron_idx in range(num_neurons):
            node_positions[(layer_idx, neuron_idx)] = (x_position, y_positions[neuron_idx])

    encoding_dim = weights_encodings[0].shape[-1] + 1  # plus one for weight itself
    fig, axs = plt.subplots(4, encoding_dim // 4 + 1, figsize=(15, 10))

    # Draw neurons and weights
    for subplot_ind in range(encoding_dim):
        ax = axs[int(subplot_ind // 2), int(subplot_ind % 2)]
        for layer_idx in range(1, len(neurons_per_layer)):
            for prev_neuron in range(neurons_per_layer[layer_idx - 1]):
                for next_neuron in range(neurons_per_layer[layer_idx]):
                    src = node_positions[(layer_idx - 1, prev_neuron)]
                    dst = node_positions[(layer_idx, next_neuron)]
                    midpoint = ((src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2)

                    if subplot_ind == encoding_dim - 1:
                        weight = weights[layer_idx - 1][next_neuron, prev_neuron]
                        color = cmap(norm(weight))  # Map this encoding dimension's value to color
                        ax.text(
                            midpoint[0],
                            midpoint[1],
                            f"{weight:.2f}",
                            color="black",
                            ha="center",
                            va="center",
                        )
                    else:
                        weight = weights_encodings[layer_idx - 1][next_neuron, prev_neuron, :]
                        color = cmap(
                            norm(weight[subplot_ind])
                        )  # Map this encoding dimension's value to color
                        ax.text(
                            midpoint[0],
                            midpoint[1],
                            f"{weight[subplot_ind]:.2f}",
                            color="black",
                            ha="center",
                            va="center",
                        )
                    ax.plot([src[0], dst[0]], [src[1], dst[1]], color=color, linewidth=2)

        # Draw neurons
        for pos in node_positions.values():
            ax.scatter(*pos, color="k")
        # ax.set_aspect('equal')
        # plt.axis('off')
    cbar_ax = fig.add_axes([0.55, 0.15, 0.4, 0.1])  # [left, bottom, width, height]
    fig.colorbar(scalar_mappable, cax=cbar_ax, orientation="horizontal")

    plt.savefig(save_name)
    plt.close()
    # plt.show()


def highlight_neighbor_weights(
    neurons_per_layer, h_layer_idx, h_neuron_idx_back, h_neuron_idx_forward
):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")  # Colormap  # noqa: F841

    # Position nodes and plot weights
    node_positions = {}  # To store positions of neurons for plotting
    y_gap = 0.1  # Vertical gap between layers  # noqa: F841
    x_gap = 0.2  # Horizontal gap within layers

    # Calculate positions for neurons
    for layer_idx, num_neurons in enumerate(neurons_per_layer):
        x_position = layer_idx * x_gap
        y_positions = np.linspace(0, 1, num_neurons + 2)[1:-1]
        for neuron_idx in range(num_neurons):
            node_positions[(layer_idx, neuron_idx)] = (x_position, y_positions[neuron_idx])

    fig, ax = plt.subplots(1, figsize=(6, 4))
    replacement_color = None
    # Draw neurons and weights
    for layer_idx in range(1, len(neurons_per_layer)):
        for i in range(neurons_per_layer[layer_idx - 1]):
            for j in range(neurons_per_layer[layer_idx]):
                if (
                    h_layer_idx == layer_idx
                    and j == h_neuron_idx_forward
                    and i != h_neuron_idx_back
                ):
                    replacement_color = "red"
                elif (
                    h_layer_idx == layer_idx
                    and j != h_neuron_idx_forward
                    and i == h_neuron_idx_back
                ):
                    replacement_color = "blue"
                elif (
                    h_layer_idx == layer_idx
                    and j == h_neuron_idx_forward
                    and i == h_neuron_idx_back
                ):
                    replacement_color = "green"
                src = node_positions[(layer_idx - 1, i)]
                dst = node_positions[(layer_idx, j)]
                color = "black"
                if replacement_color is not None:
                    color = replacement_color
                    ax.plot([src[0], dst[0]], [src[1], dst[1]], color=color, linewidth=4)
                    replacement_color = None
                else:
                    ax.plot([src[0], dst[0]], [src[1], dst[1]], color=color, linewidth=2)

        # Draw neurons
        for pos in node_positions.values():
            ax.scatter(*pos, color="k")
        # ax.set_aspect('equal')
        # plt.axis('off')
    plt.savefig("weight_neighbors.png")
    plt.close()
    # plt.show()


def highlight_indexed_weights(neurons_per_layer, highlighted_weight_coordinates):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")  # Colormap  # noqa: F841

    # Position nodes and plot weights
    node_positions = {}  # To store positions of neurons for plotting
    y_gap = 0.1  # Vertical gap between layers  # noqa: F841
    x_gap = 0.2  # Horizontal gap within layers

    # Calculate positions for neurons
    for layer_idx, num_neurons in enumerate(neurons_per_layer):
        x_position = layer_idx * x_gap
        y_positions = np.linspace(0, 1, num_neurons + 2)[1:-1]
        for neuron_idx in range(num_neurons):
            node_positions[(layer_idx, neuron_idx)] = (x_position, y_positions[neuron_idx])

    fig, ax = plt.subplots(1, figsize=(6, 4))
    replacement_color = None  # noqa: F841
    # Draw neurons and weights
    for layer_idx in range(1, len(neurons_per_layer)):
        for i in range(neurons_per_layer[layer_idx - 1]):
            for j in range(neurons_per_layer[layer_idx]):
                src = node_positions[(layer_idx - 1, i)]
                dst = node_positions[(layer_idx, j)]
                color = "black"
                ax.plot([src[0], dst[0]], [src[1], dst[1]], color=color, linewidth=2)
    for layer_idx, i, j in highlighted_weight_coordinates:
        src = node_positions[(layer_idx - 1, j)]
        dst = node_positions[(layer_idx, i)]
        color = "yellow"
        ax.plot([src[0], dst[0]], [src[1], dst[1]], color=color, linewidth=4)

        # Draw neurons
        for pos in node_positions.values():
            ax.scatter(*pos, color="k")
        # ax.set_aspect('equal')
        # plt.axis('off')
    plt.savefig("highlighted_weights.png")
    plt.close()
