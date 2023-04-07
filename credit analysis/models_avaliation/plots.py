import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_target_vs_score(target, scores, quantiles, lower_limit=None):
  eval_df = pd.DataFrame(zip(target, scores), columns=['TARGET', 'SCORE']).sort_values(by='SCORE')

  quantile = pd.qcut(eval_df['SCORE'], quantiles, labels=range(1,1+quantiles))
  eval_df['QUANTILE'] = quantile
  if lower_limit:
      eval_df = eval_df.loc[eval_df['QUANTILE'] > lower_limit]

  chart_df = pd.DataFrame(zip(
    eval_df.groupby('QUANTILE')['TARGET'].mean(), eval_df.groupby('QUANTILE')['TARGET'].size()/eval_df.shape[0]
    ), columns=['TAXA_DE_MAUS', 'POPULACAO'])

  fig = make_subplots(specs=[[{"secondary_y": True}]])

  chart_df = chart_df.dropna()
  
  fig.add_trace(
    go.Bar(x=chart_df.index.to_list(), y=chart_df.POPULACAO, name="Representativeness (%)", marker_color='rgba(0, 0, 87, .8)'),
    secondary_y=False,
  )

  fig.add_trace(
    go.Scatter(x=chart_df.index.to_list(), y=chart_df.TAXA_DE_MAUS, name="Weighted target mean by exposure (%)", marker_color='rgb(216, 125, 77)'),
    secondary_y=True,
    # labels=dict(x="Decil", y="Representativeness (%)")
  )

  fig.update_xaxes(title_text='Population Decile')
  fig.update_yaxes(title_text='Representativeness (%)', secondary_y=False)
  fig.update_yaxes(title_text='Weighted Target Mean by Exposure (%)', secondary_y=True)
  fig.update_layout(width=1000,height=500, legend=dict(orientation="h", xanchor='center', x=0.5, y=-0.2))

  fig.show()