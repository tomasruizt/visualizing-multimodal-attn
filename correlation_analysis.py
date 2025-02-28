import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_correlation_progression(x1, x2, n_img_tokens=256, p_threshold=0.01):
    """
    Calculate and visualize layer-wise correlations between two tensors.

    Args:
        x1: Tensor of values with shape [layers, tokens]
        x2: Tensor of values with shape [layers, tokens]
        n_img_tokens: Number of image tokens to analyze (default: 256)

    Returns:
        tuple: (layer_correlations, layer_pvalues) - Lists of correlation values and p-values
    """
    # Calculate layer-wise correlations
    layer_correlations = []
    layer_pvalues = []

    for layer in range(x1.shape[0]):
        # Get data for this layer
        x1_img_toks = x1[layer, :n_img_tokens].numpy()
        x2_img_toks = x2[layer, :n_img_tokens].numpy()

        # Calculate Spearman correlation for this layer
        corr, pval = stats.spearmanr(x1_img_toks, x2_img_toks)
        # Handle NaN correlation values (occurs with constant arrays)
        if np.isnan(corr):
            corr = 0.0  # Replace NaN with zero correlation
        layer_correlations.append(corr)
        layer_pvalues.append(pval)

    # Plot the layer-wise correlations
    plt.figure(figsize=(5, 3))
    plt.plot(
        range(len(layer_correlations)), layer_correlations, "o-", label="Correlation"
    )
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.3)
    plt.xlabel("Layer")
    plt.ylabel("Spearman Correlation")
    plt.grid(True)

    # Add significance markers
    sig_layers = np.array(layer_pvalues) < p_threshold
    plt.plot(
        np.where(sig_layers)[0],
        np.array(layer_correlations)[sig_layers],
        "r*",
        label=f"p < {p_threshold}",
        markersize=10,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nLayer-wise correlation summary:")
    print(
        f"Max correlation: {max(layer_correlations):.3f} (Layer {np.argmax(layer_correlations)})"
    )
    print(
        f"Min correlation: {min(layer_correlations):.3f} (Layer {np.argmin(layer_correlations)})"
    )
    print(f"Mean correlation: {np.mean(layer_correlations):.3f}")
    print(
        f"Number of significant correlations (p < {p_threshold}): {sum(np.array(layer_pvalues) < p_threshold)}/{len(layer_correlations)}"
    )

    return layer_correlations, layer_pvalues


def plot_correlation_in_midlayers(
    x1, x2, start_layer=8, end_layer=16, n_img_tokens=256, n_cols=3
):
    """
    Plot layer-wise correlations between two tensors for specified layers.

    Args:
        x1: First tensor with shape [layers, tokens]
        x2: Second tensor with shape [layers, tokens]
        start_layer: First layer to analyze (inclusive)
        end_layer: Last layer to analyze (inclusive)
        n_img_tokens: Number of image tokens to analyze
        n_cols: Number of columns in the subplot grid
    """
    middle_layers = range(start_layer, end_layer + 1)

    # Create a figure with subplots for each layer
    n_layers = end_layer - start_layer + 1
    n_rows = (n_layers + n_cols - 1) // n_cols  # Ceiling division for number of rows

    plt.figure(figsize=(8, 3 * n_rows))

    for idx, layer in enumerate(middle_layers):
        # Get data for this layer
        x1_img_toks = x1[layer, :n_img_tokens].numpy()
        x2_img_toks = x2[layer, :n_img_tokens].numpy()

        # Calculate Spearman correlation for this layer
        corr, pval = stats.spearmanr(x1_img_toks, x2_img_toks)

        # Create subplot
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.scatter(x1_img_toks, x2_img_toks, alpha=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True)

        # Add correlation information to title
        title = f"Layer {layer}\n corr={corr:.3f}"
        if pval < 0.05:
            title += " *"
        plt.title(title)

        # Only add labels on the left and bottom edges
        if idx % n_cols == 0:  # leftmost plots
            plt.ylabel("Causal Probabilities")
        if idx >= n_layers - n_cols:  # bottom plots
            plt.xlabel("Normalized Importance")

    plt.tight_layout()
    plt.show()
