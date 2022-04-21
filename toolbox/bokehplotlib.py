from pathlib import Path
import functools

import pandas as pd
from bokeh.io import save
from bokeh.layouts import gridplot as gp
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource



class PWrapper:

    def __init__(self, p):
        self.p = p
        self.fontsize = 18

    @property
    def layoutDOM(self):
        return self.p

    def show(self):
        show(self.p)

    def show_notebook(self):
        pass

    def title(self, title: str, fontsize=None):
        self.p.title.text = title
        fontsize =  fontsize if fontsize is not None else self.fontsize
        fontsize = f'{fontsize}px'
        self.p.title.text_font_size = fontsize


    def xlabel(self, xlabel: str, fontsize=None):
        pass

    def ylabel(self, ylabel: str, fontsize=None):
        pass

    def savefig(self, filename):
        save(self.p, filename=filename)

    def scatter(self, *args, **kwargs):
        self.p = scatter(p=self.p.layoutDOM, *args, **kwargs)

    def __str__(self):
        return """
        # Tip1: X軸ラベルとその回転
          p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
          p.xaxis.major_label_orientation = 1.2
        """


def gridplot(figures: list, cols=3):

    def split_list(l, n):
        for idx in range(0, len(l), n):
            yield l[idx:idx + n]

    figures = [p.layoutDOM for p in figures]
    _figures = list(split_list(figures, cols))
    p = gp(_figures, toolbar_location="right")
    p = PWrapper(p)
    return p


def plot(data: pd.DataFrame, x: str, y: str,
         hue=None, s=None, c=None, figsize=(400, 300),
         mode="scatter", p=None):

    source = ColumnDataSource(data)

    if p is None:
        p = figure(width=figsize[0], height=figsize[1],
                   tools="pan,wheel_zoom,box_zoom,box_select,hover,reset,save")

    c = c if c is not None else "steelblue"
    s = s if s is not None else 5

    #: scatter
    if mode == "scatter":
        if hue is None:
            p.circle(x, y, source=source, color=c, size=s, alpha=0.9)
        else:
            p.circle(x, y, source=source, color=hue, size=s, alpha=0.9)

    elif mode == "bar":
        p.vbar(x='fruits', top='counts', width=0.9, color='color', legend_field="fruits", source=source)

    else:
        raise NotImplementedError(mode)

    p = PWrapper(p)

    return p


scatter = functools.partial(plot, mode="scatter")
hist = functools.partial(plot, y=None, mode="hist")
bar = functools.partial(plot, y=None, mode="bar")




if __name__ == '__main__':
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

    """
    species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex
    """
    import copy
    p1 = scatter(df, x="bill_length_mm", y="bill_depth_mm", hue="species")
    p1.title("this is title", fontsize=18)
    p2 = scatter(df, x="bill_length_mm", y="bill_depth_mm", hue="species")
    p3 = scatter(df, x="bill_length_mm", y="bill_depth_mm", hue="species")

    p = gridplot([p1, p2, p3], cols=1)
    p.show()
    #p = hist(df, x="bill_length_mm", hue="species")
