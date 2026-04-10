import random
import textwrap

import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx

from filtering import ThreadFilter
from interfaces import InterventionResult, ReasoningResult, RedditThread
from models import ToxicityClassifier
from orchestrator import ModerationOrchestrator


class DummyReasoner:
    def analyze_intent(
        self, text: str, parent_text: str, root_context: str
    ) -> ReasoningResult:
        roll = random.random()
        if roll > 0.9:
            return {
                "category": "zero-tolerance",
                "points": 10,
                "explanation": "Severe attack.",
            }
        elif roll > 0.4:
            return {
                "category": "toxic",
                "points": random.randint(2, 5),
                "explanation": "Hostile tone.",
            }
        else:
            return {
                "category": "flare",
                "points": 0,
                "explanation": "Just heated debate.",
            }


class DummyIntervener:
    def generate_intervention(
        self, text: str, author: str, infractions: int
    ) -> InterventionResult:
        return {
            "intervention_text": "Please keep the discussion civil.",
            "tone_used": "neutral",
        }


def visualize_graph(G: nx.DiGraph) -> None:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    pos = nx.spring_layout(  # pyright: ignore[reportUnknownVariableType]
        G,                   # pyright: ignore[reportUnknownArgumentType]
        k=1.2,
        iterations=100,
        seed=42,
    )

    edge_x = []
    edge_y = []
    for edge in G.edges():  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # pyright: ignore[reportUnknownMemberType]
        edge_y.extend([y0, y1, None])  # pyright: ignore[reportUnknownMemberType]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    text_labels = []
    hover_texts = []

    cmap = plt.get_cmap("Reds")

    for node, data in G.nodes(data=True):  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAny]
        x, y = pos[node]
        node_x.append(x)  # pyright: ignore[reportUnknownMemberType]
        node_y.append(y)  # pyright: ignore[reportUnknownMemberType]

        score: float = data.get("toxicity_score", 0.0)  # pyright: ignore[reportAny]
        author: str = data.get("author", "Unknown")     # pyright: ignore[reportAny]
        
        content: str = data.get("text", "") # pyright: ignore[reportAny]
        
        wrapped_text = "<br>".join(textwrap.wrap(content, width=60))

        if data.get("type") == "post":  # pyright: ignore[reportAny]
            node_colors.append("gold")  # pyright: ignore[reportUnknownMemberType]
            node_sizes.append(55)  # pyright: ignore[reportUnknownMemberType]
            text_labels.append(f"Original Post<br>{author}")  # pyright: ignore[reportUnknownMemberType]
            hover_texts.append(f"<b>Original Post</b><br>Author: {author}<br><br>{wrapped_text}")  # pyright: ignore[reportUnknownMemberType]
        
        else:
            node_colors.append(mcolors.to_hex(cmap(score)))  # pyright: ignore[reportUnknownMemberType]
            node_sizes.append(45)  # pyright: ignore[reportUnknownMemberType]
            
            display_author = author
            if len(author) > 8:
                display_author = author[:5] + "..."
                
            text_labels.append(f"{display_author}<br>Tox: {score:.2f}")  # pyright: ignore[reportUnknownMemberType]
            hover_texts.append(f"<b>Comment</b><br>Author: {author}<br>Toxicity: {score:.2f}<br><br>{wrapped_text}")  # pyright: ignore[reportUnknownMemberType]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=hover_texts,
        text=text_labels,
        textposition="bottom center",
        marker=dict(
            color=node_colors,  # pyright: ignore[reportUnknownArgumentType]
            size=node_sizes,  # pyright: ignore[reportUnknownArgumentType]
            line=dict(color='black', width=1.5)
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Thread Result",
                font=dict(size=16),
                x=0.5
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    fig.show()  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    classifier = ToxicityClassifier()

    # 1. Filter JSONL files
    filter_engine = ThreadFilter(classifier, threshold=0.1, max_threads=2)
    target_threads: list[RedditThread] = filter_engine.filter_file("data.jsonl")

    # 2. Run Orchestrator
    orchestrator = ModerationOrchestrator(
        classifier, reasoner=DummyReasoner(), intervener=DummyIntervener()
    )

    for thread in target_threads:
        print(f"\n--- Processing Thread: {thread['title']} ---")
        processed_graph = orchestrator.process_thread(thread)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        visualize_graph(processed_graph)
