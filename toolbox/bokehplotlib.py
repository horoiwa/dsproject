from pathlib import Path
import functools

import numpy as np
import pandas as pd
from bokeh.io import save, output_notebook
from bokeh.layouts import gridplot as gp
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CategoricalColorMapper, Legend
from bokeh.palettes import Category10, Category20


class PWrapper:

    def __init__(self, p):
        self.p = p
        self.fontsize = 18

    @property
    def layoutDOM(self):
        return self.p

    def show(self):
        show(self.p)

    def title(self, title: str, fontsize=None):
        self.p.title.text = title
        fontsize =  fontsize if fontsize is not None else self.fontsize
        fontsize = f'{fontsize}px'
        self.p.title.text_font_size = fontsize


    def xlabel(self, xlabel: str, fontsize=None):
        self.p.xaxis.axis_label = xlabel
        if fontsize is not None:
            self.p.xaxis.axis_label_text_font_size = f'{fontsize}px'

    def ylabel(self, ylabel: str, fontsize=None):
        self.p.yaxis.axis_label = ylabel
        if fontsize is not None:
            self.p.yaxis.axis_label_text_font_size = f'{fontsize}px'

    def legend(self, inline=False):
        self.p.legend.visible = True
        self.p.legend.background_fill_alpha = 0.3
        if not inline:
            self.p.add_layout(self.p.legend[0], 'right')

    def savefig(self, filename):
        save(self.p, filename=filename)

    def scatter(self, *args, **kwargs):
        self.p = scatter(p=self.p.layoutDOM, *args, **kwargs)

    def __str__(self):
        return """
        # Tip: 軸ラベルとその回転
          p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
          p.xaxis.major_label_orientation = 1.2
        #: Tip: グリッド表示
          p.xgrid.grid_line_color = None
        #: Tip: ラベルの調整
          p.legend.orientation = "horizontal"
          p.legend.location = "top_center"
        """


def toColumnDataSource(df, c=None, hue=None, cmap=None):

    if hue is not None:
        categories = df[hue].unique().tolist()
        n_cats = len(categories)
        if cmap is None:
            cmap = Category10[10] if n_cats <= 10 else Category20[20]

        num_colors = len(cmap)
        colormap = {cat: cmap[i%num_colors] for i, cat in enumerate(categories)}
        colors = [colormap[v] for v in df[hue]]
    else:
        colors = c if c is not None else "steelblue"

    df["fill_color"] = colors

    source = ColumnDataSource(df)
    return source


def scatter(df: pd.DataFrame, x: str, y: str,
            hue=None, s=8, c=None, figsize=(500, 400),
            p=None, cmap=None):


    if p is None:
        p = figure(width=figsize[0], height=figsize[1],
                   tools="pan,wheel_zoom,box_zoom,box_select,hover,reset,save")

    source = toColumnDataSource(df, c, hue, cmap)

    #: scatter
    if hue is not None:
        p.circle(x, y, source=source, color="fill_color", size=s, alpha=0.9, legend_group=hue)
        p.legend.visible = False
    else:
        p.circle(x, y, source=source, color="fill_color", size=s, alpha=0.9)

    p = PWrapper(p)
    p.xlabel(x)
    p.ylabel(y)

    return p


def hist(df, x, bins=30, density=True, hue=None, c=None,
         figsize=(500, 400), p=None, cmap=None):

    if p is None:
        p = figure(width=figsize[0], height=figsize[1],
                   tools="pan,wheel_zoom,box_zoom,box_select,hover,reset,save")

    if hue is None:
        data = df[x].dropna().values
        hist, edges = np.histogram(data, density=density, bins=bins)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
    else:
        groups = df.groupby(hue)
        if cmap is None:
            cmap = Category10[10] if len(groups) <= 10 else Category20[20]

        for i, (name, df_group) in enumerate(groups):
            data = df_group[x].dropna().values
            hist, edges = np.histogram(data, density=density, bins=bins)
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                   line_color="white", color=cmap[i], alpha=0.5, legend_label=name)


    p = PWrapper(p)
    p.xlabel(x)

    return p


def gridplot(figures: list, cols=3):

    def split_list(l, n):
        for idx in range(0, len(l), n):
            yield l[idx:idx + n]

    figures = [p.layoutDOM for p in figures]
    _figures = list(split_list(figures, cols))
    p = gp(_figures, toolbar_location="right")
    p = PWrapper(p)
    return p


def facetplot(df: pd.DataFrame, x, y, col, row=None, hue=None):

    if col and row:
        df = df.dropna(subset=[row, col], axis=0)
        groups = df.groupby([row, col])
    else:
        df = df.dropna(subset=[col], axis=0)
        groups = df.groupby(col)

    figs = []
    for (name, df_group) in groups:
        name = name[0] + " | " + name[1] if type(name) == tuple else name
        p = scatter(df_group, x,  y, hue=hue)
        p.title(name)
        if hue is not None:
            p.legend(inline=True)
        figs.append(p)

    p = gridplot(figs, cols=df[col].nunique())
    return p



if __name__ == '__main__':
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

    """
    species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex
    """
    p1 = scatter(df, x="bill_length_mm", y="bill_depth_mm", hue="species")
    p1.title("this is title", fontsize=18)
    p1.xlabel("xlabel", fontsize=18)
    p1.ylabel("ylabel")
    p1.legend()

    p2 = hist(df, x="bill_length_mm", c="olive")
    p3 = hist(df, x="bill_length_mm", hue="island")

    p = gridplot([p1, p2, p3], cols=3)
    #p.show()

    p4 = facetplot(df, x="bill_length_mm", y="bill_depth_mm", col="sex", hue="species")
    #p4 = facetplot(df, x="bill_length_mm", y="bill_depth_mm", col="sex", row="island", hue="species")
    p4.show()
