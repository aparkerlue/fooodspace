import itertools
import matplotlib.pyplot as plt
import numpy as np

from bokeh.io import output_file, show
from bokeh.models import (
    GMapPlot, GMapOptions, ColumnDataSource, Circle, Range1d,
    PanTool, BoxZoomTool, BoxSelectTool, WheelZoomTool, ResetTool, SaveTool,
)

from fooodspace.data import yelp_businesses_to_dataframe
from fooodspace.settings import GOOGLE_API_KEY


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    From:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Observed values')
    plt.xlabel('Predicted values')


def plot_restaurant_map(data, y):
    df = yelp_businesses_to_dataframe(data, y)

    plot = GMapPlot(
        api_key=GOOGLE_API_KEY,
        x_range=Range1d(),
        y_range=Range1d(),
        map_options=GMapOptions(
            lat=40.735, lng=-73.990, map_type="roadmap", zoom=13,
        ),
    )
    plot.toolbar.logo = None
    plot.toolbar_location = 'above'
    plot.title.text = "NYC restaurant sample ({:,})".format(len(df))
    plot.add_tools(
        PanTool(), BoxZoomTool(), BoxSelectTool(),
        WheelZoomTool(), ResetTool(), SaveTool(),
    )

    source = ColumnDataSource(
        data=df[['lat', 'lon', 'catcolor', 'revsize']],
    )
    circle = Circle(
        x="lon", y="lat",
        line_color='black',
        fill_color='catcolor',
        fill_alpha=0.8,
        size='revsize',
    )
    plot.add_glyph(source, circle)

    output_file("plots/restaurant_plot.html")
    show(plot)
