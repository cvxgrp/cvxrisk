import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    import numpy as np

    from cvx.risk import Model

    return Model, np


if __name__ == "__main__":
    app.run()
