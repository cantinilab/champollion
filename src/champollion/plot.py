import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

__all__ = [
    "plot_aggregated_transport_plan",
    "plot_aggregated_cost_matrix",
    "top_interactions_bar",
    "get_plot_top_interactions",
]


def _aggregate_by_annotation(
    heat_mtx,
    annotations,
    annotations_ordered,
    annotations_2=None,
    annotations_ordered_2=None,
    reduction="median",
):
    annotations = np.asarray(annotations)
    annotations_2 = annotations if annotations_2 is None else np.asarray(annotations_2)
    annotations_ordered_2 = (
        annotations_ordered if annotations_ordered_2 is None else annotations_ordered_2
    )

    row_groups = [np.nonzero(annotations == value)[0] for value in annotations_ordered]
    col_groups = [
        np.nonzero(annotations_2 == value)[0] for value in annotations_ordered_2
    ]

    aggregated = np.zeros((len(row_groups), len(col_groups)))
    for i, idxs_1 in enumerate(row_groups):
        for j, idxs_2 in enumerate(col_groups):
            block = heat_mtx[idxs_1, :][:, idxs_2]
            if reduction == "sum":
                aggregated[i, j] = np.nansum(block) / min(len(idxs_1), len(idxs_2))
            elif reduction == "median":
                aggregated[i, j] = np.nanmedian(block)
            else:
                raise ValueError("reduction must be 'sum' or 'median'.")
    return aggregated, annotations_ordered, annotations_ordered_2


def _plot_aggregated_matrix(
    heat_mtx,
    annotations,
    annotations_ordered,
    annotations_2=None,
    annotations_ordered_2=None,
    cbar_label="transport mass",
    save_path=None,
    reduction="median",
    figsize=(6, 6),
    x_rotation=45,
    x_ha="right",
):
    """Plot a cell-level matrix aggregated by cell annotations (e.g. cell types).

    Parameters
    ----------
    heat_mtx
        Cell-level matrix with cells from modality 1 on rows and cells from
        modality 2 on columns.
    annotations
        Annotation labels for the rows of ``heat_mtx``. For example, cell
        types.
    annotations_ordered
        Annotation values to show on the y-axis, in plotting order.
    annotations_2
        Annotation labels for the columns of ``heat_mtx``. If ``None``, the row
        labels are reused.
    annotations_ordered_2
        Annotation values to show on the x-axis, in plotting order. If ``None``,
        the row annotation order is reused.
    cbar_label
        Label shown next to the colorbar.
    save_path
        Optional path where the figure should be saved.
    reduction
        Aggregation used within each cell-type pair. ``"sum"`` computes the
        total block mass divided by the smaller cell-type block size.
        ``"median"`` uses the median value in each block.
    figsize
        Matplotlib figure size.
    x_rotation
        Rotation angle for x-axis tick labels.
    x_ha
        Horizontal alignment for x-axis tick labels.

    Returns
    -------
    aggregated
        Annotation-by-annotation aggregated matrix.
    fig
        Matplotlib figure.
    ax
        Matplotlib axes.
    """
    aggregated, annotations_ordered, annotations_ordered_2 = _aggregate_by_annotation(
        heat_mtx=heat_mtx,
        annotations=annotations,
        annotations_ordered=annotations_ordered,
        annotations_2=annotations_2,
        annotations_ordered_2=annotations_ordered_2,
        reduction=reduction,
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(aggregated)
    ax.set_xticks(
        np.arange(len(annotations_ordered_2)),
        annotations_ordered_2,
        rotation=x_rotation,
        ha=x_ha,
    )
    ax.set_yticks(np.arange(len(annotations_ordered)), annotations_ordered, rotation=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, location="right")
    cbar.set_label(cbar_label)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return aggregated, fig, ax


def plot_aggregated_transport_plan(
    heat_mtx,
    annotations,
    annotations_ordered,
    annotations_2=None,
    annotations_ordered_2=None,
    cbar_label="transport mass",
    save_path=None,
    reduction="sum",
    figsize=(6, 6),
    x_rotation=45,
    x_ha="right",
):
    """Plot a transport plan aggregated by cell annotations (e.g. cell types).

    The transport plan is sum-aggregated block-wise, which can help visualize whether
    transported mass between annotated cell populations makes sense.

    Parameters
    ----------
    heat_mtx
        Cell-level transport matrix with cells from modality 1 on rows and cells
        from modality 2 on columns.
    annotations
        Annotation labels for the rows of ``heat_mtx``. For example, cell
        types.
    annotations_ordered
        Annotation values to show on the y-axis, in plotting order.
    annotations_2
        Annotation labels for the columns of ``heat_mtx``. If ``None``, the row
        labels are reused.
    annotations_ordered_2
        Annotation values to show on the x-axis, in plotting order. If ``None``,
        the row annotation order is reused.
    cbar_label
        Label shown next to the colorbar.
    save_path
        Optional path where the figure should be saved.
    reduction
        Aggregation applied within each annotation pair. With ``"sum"``, each
        block is summed and divided by the smaller block size.
    figsize
        Matplotlib figure size.
    x_rotation
        Rotation angle for x-axis tick labels.
    x_ha
        Horizontal alignment for x-axis tick labels.

    Returns
    -------
    aggregated
        Annotation-by-annotation aggregated transport matrix.
    fig
        Matplotlib figure.
    ax
        Matplotlib axes.
    """
    return _plot_aggregated_matrix(
        heat_mtx=heat_mtx,
        annotations=annotations,
        annotations_ordered=annotations_ordered,
        annotations_2=annotations_2,
        annotations_ordered_2=annotations_ordered_2,
        cbar_label=cbar_label,
        save_path=save_path,
        reduction=reduction,
        figsize=figsize,
        x_rotation=x_rotation,
        x_ha=x_ha,
    )


def plot_aggregated_cost_matrix(
    heat_mtx,
    annotations,
    annotations_ordered,
    annotations_2=None,
    annotations_ordered_2=None,
    cbar_label="cost",
    save_path=None,
    reduction="median",
    figsize=(6, 6),
    x_rotation=45,
    x_ha="right",
):
    """Plot a cost matrix aggregated by cell annotations (e.g. cell types).

    The cost matrix is median-aggregated block-wise, which can help visualize whether
    relative inferred costs between annotated cell populations make sense.

    Parameters
    ----------
    heat_mtx
        Cell-level cost matrix with cells from modality 1 on rows and cells
        from modality 2 on columns.
    annotations
        Annotation labels for the rows of ``heat_mtx``. For example, cell
        types.
    annotations_ordered
        Annotation values to show on the y-axis, in plotting order.
    annotations_2
        Annotation labels for the columns of ``heat_mtx``. If ``None``, the row
        labels are reused.
    annotations_ordered_2
        Annotation values to show on the x-axis, in plotting order. If ``None``,
        the row annotation order is reused.
    cbar_label
        Label shown next to the colorbar.
    save_path
        Optional path where the figure should be saved.
    reduction
        Aggregation applied within each annotation pair. With ``"median"``, each
        block is summarized by its median value.
    figsize
        Matplotlib figure size.
    x_rotation
        Rotation angle for x-axis tick labels.
    x_ha
        Horizontal alignment for x-axis tick labels.

    Returns
    -------
    aggregated
        Annotation-by-annotation aggregated cost matrix.
    fig
        Matplotlib figure.
    ax
        Matplotlib axes.
    """
    return _plot_aggregated_matrix(
        heat_mtx=heat_mtx,
        annotations=annotations,
        annotations_ordered=annotations_ordered,
        annotations_2=annotations_2,
        annotations_ordered_2=annotations_ordered_2,
        cbar_label=cbar_label,
        save_path=save_path,
        reduction=reduction,
        figsize=figsize,
        x_rotation=x_rotation,
        x_ha=x_ha,
    )


def top_interactions_bar(interactions, title=None, colors=None):
    """Plot a horizontal bar chart of top feature interaction scores.

    Parameters
    ----------
    interactions
        DataFrame returned by :meth:`champollion.Champollion.top_interactions`.
        Must contain ``"target_feature"`` and ``"weight"`` columns.
    title
        Plot title. If ``None``, a generic title is used.
    colors
        Matplotlib-compatible color or sequence of colors. If ``None``, a red
        sequential palette is used.

    Returns
    -------
    fig
        Matplotlib figure.
    ax
        Matplotlib axes.
    """
    required_columns = {"target_feature", "weight"}
    missing_columns = required_columns - set(interactions.columns)
    if missing_columns:
        raise ValueError(
            "interactions must contain columns "
            f"{sorted(required_columns)}; missing {sorted(missing_columns)}."
        )
    df = interactions.loc[:, ["target_feature", "weight"]].rename(
        columns={"target_feature": "feature", "weight": "score"}
    )
    df = df.sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    if colors is None:
        colors = sns.color_palette("Reds", n_colors=len(df))
    ax.barh(df["feature"], df["score"], color=colors)
    ax.set_xlabel("A", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Top interactions" if title is None else title, fontsize=13)
    sns.despine(ax=ax)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    return fig, ax


def get_plot_top_interactions(
    model,
    feature,
    modality,
    k=10,
    direction="both",
    title=None,
    colors=None,
):
    """Get and plot top inferred interactions for one feature directly.

    Parameters
    ----------
    model
        Fitted :class:`champollion.Champollion` model used to retrieve the
        fit interaction matrix.
    feature
        Feature name passed to :meth:`champollion.Champollion.top_interactions`.
    modality
        Modality containing ``feature``.
    k
        Maximum number of non-zero interactions to request from the model.
    direction
        ``"positive"``, ``"negative"``, or ``"both"``. Passed through to
        :meth:`champollion.Champollion.top_interactions`.
    title
        Plot title. If ``None``, a title is generated from ``feature`` and
        ``direction``.
    colors
        Matplotlib-compatible color or sequence of colors. If ``None``, a red
        sequential palette is used.

    Returns
    -------
    fig
        Matplotlib figure.
    ax
        Matplotlib axes.
    """
    interactions = model.top_interactions(
        feature=feature,
        modality=modality,
        k=k,
        direction=direction,
    )
    if title is None:
        if direction == "both":
            title = f"Top interactions with {feature}"
        else:
            title = f"Top {direction} interactions with {feature}"
    return top_interactions_bar(interactions=interactions, title=title, colors=colors)
