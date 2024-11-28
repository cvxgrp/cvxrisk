import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    import numpy as np

    return (np,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
