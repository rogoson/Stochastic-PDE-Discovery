from tabulate import tabulate
from IPython.core.display import display_html


def tabulate_neatly(table, headers=None, title=None, **kwargs):
    if title is not None:
        display_html(f"<h3>{title}</h3>\n", raw=True)
    display_html(tabulate(table, headers=headers, tablefmt="html", **kwargs))
