import click
from hooks.create_features import FUNCS


@click.group()
def cli():
    pass

@cli.command()
@click.option("--out", default="data/raw/raw.csv", type=str)
def loaddata(out):
    df = None
    print(FUNCS)
    pass


@cli.command()
@click.option("--target", default="data/raw/raw.csv", type=str)
@click.option("--out", default="data/crude/crude.csv", type=str)
@click.option("--poly", default=2, type=int)
def create_features(target, out):
    pass


@cli.command()
@click.option("--target", default="data/crude/crude.csv", type=str)
@click.option("--out", default="data/refined/refined.csv", type=str)
def screening(target, out):
    pass


if __name__ == "__main__":
    cli()
