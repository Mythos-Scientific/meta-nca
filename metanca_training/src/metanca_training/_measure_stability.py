import jax
import jax.numpy as jnp


def measure_stability(
    loss_values: jax.Array, epsilon_proportion: float = 0.1
) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Measures the stability of a learning algorithm based on the longest contiguous period
    where the loss stays within a defined basin around the minimum loss.

    Arguments
    ---------
    loss_values: ArrayLike
        Loss-values per epoch
    epsilon_proportion: float
        The proportion to define the tolerance level around the minimum loss,
        i.e., 0.01 is a bound of 1% of the loss value.

    Returns
    -------
    tuple[int, list[utple[int, int]]]
        length of the longest contiguous stable period,
        all stable periods with their start and end epochs.
    """
    # Ensure loss values are a numpy array for easier processing
    loss_values = jnp.array(loss_values)

    # Step 1: Find the minimum loss
    L_min = jnp.min(loss_values)

    # Step 2: Define the basin threshold (epsilon)
    epsilon = epsilon_proportion * L_min

    # Step 3: Identify epochs in the basin
    epochs_in_basin = [i for i, loss in enumerate(loss_values) if loss <= L_min + epsilon]

    # Step 4: Find the longest contiguous sequence in the basin
    def longest_contiguous_sequence(epochs):
        if not epochs:
            return 0
        longest, current = 1, 1
        for i in range(1, len(epochs)):
            if epochs[i] == epochs[i - 1] + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest

    longest_stable_period = longest_contiguous_sequence(epochs_in_basin)

    # Step 5: Identify all contiguous sequences
    def all_contiguous_sequences(epochs):
        if not epochs:
            return []
        sequences = []
        current_seq = [epochs[0]]
        for i in range(1, len(epochs)):
            if epochs[i] == epochs[i - 1] + 1:
                current_seq.append(epochs[i])
            else:
                sequences.append((current_seq[0], current_seq[-1], len(current_seq)))
                current_seq = [epochs[i]]
        sequences.append((current_seq[0], current_seq[-1], len(current_seq)))
        return sequences

    all_stable_periods = all_contiguous_sequences(epochs_in_basin)

    # Output the results
    return longest_stable_period, all_stable_periods
