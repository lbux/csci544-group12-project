# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false
from __future__ import annotations

import textwrap

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import plotly.graph_objects as go
from networkx.drawing.nx_pydot import graphviz_layout


def visualize_graph(G: nx.DiGraph[str]) -> None:
    """Produces a graph with dynamic data from a selected Reddit thread."""
    G_layout = nx.DiGraph()
    G_layout.add_nodes_from(G.nodes())
    G_layout.add_edges_from(G.edges())

    pos = graphviz_layout(G_layout, prog="dot")

    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    text_labels = []
    hover_texts = []

    cmap = cm.get_cmap("Reds")

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        score: float = data.get("toxicity_score", 0.0)
        author: str = data.get("author", "Unknown")
        content: str = data.get("body", "")

        wrapped_text = "<br>".join(textwrap.wrap(content, width=60))

        if data.get("type") == "post":
            node_colors.append("gold")
            node_sizes.append(55)
            text_labels.append(f"Original Post<br>{author}")
            hover_texts.append(
                f"<b>Original Post</b><br>Author: {author}<br><br>{wrapped_text}"
            )
        else:
            node_colors.append(mcolors.to_hex(cmap(score)))
            node_sizes.append(45)

            display_author = author
            if len(author) > 8:
                display_author = author[:5] + "..."

            text_labels.append(f"{display_author}<br>Tox: {score:.2f}")
            hover_texts.append(
                f"<b>Comment</b><br>Author: {author}<br>Toxicity: {score:.2f}<br><br>{wrapped_text}"
            )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=hover_texts,
        text=text_labels,
        textposition="bottom center",
        marker=dict(
            color=node_colors, size=node_sizes, line=dict(color="black", width=1.5)
        ),
    )

    edge_annotations = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_annotations.append(
            dict(
                ax=x0,
                ay=y0,
                axref="x",
                ayref="y",
                x=x1,
                y=y1,
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=1.5,
                arrowcolor="gray",
                standoff=25,
            )
        )

    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            title=dict(text="Thread Result", font=dict(size=16), x=0.5),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            annotations=edge_annotations,
        ),
    )

    fig.show()
