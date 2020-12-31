import ipywidgets as ip
import plotly.graph_objects as go
import pandas as pd

data = pd.read_csv('total_embs.tsv', sep='\t')
fig = go.Figure(data=go.Scatter(
                            x=data['x'],
                            y=data['y'],
                            mode='markers',
                            text=data['Name']
))
fig.update_layout(title='Sprite Suite Character Embeddings')
fig.show()