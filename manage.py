from datetime import datetime
from pathlib import Path
import shutil

import pytz
import click


@click.group()
def cli():
    pass

@cli.command()
def startproject():

    project_dir = Path(get_dirname())

    if project_dir.exists():
        ans = input(f"{project_dir} already exists, Overwrite? [yes/no]: ").lower()
        if ans not in ["yes", "no"]:
            raise Exception(f"Invalid input: {val}")

        if ans == "yes":
            shutil.rmtree(project_dir)
        elif ans == "no":
            n = 1
            while project_dir.exists():
                project_dir = Path(str(project_dir).split("-")[0] + f"-{n}")
                n += 1

    click.echo(f"Create directory: {project_dir}")
    project_dir.mkdir()


@cli.command()
def collectstatic():
    pass


def get_dirname():
    tz = pytz.timezone('Japan')
    dt = tz.localize(datetime.today())
    dirname = "proj_" + dt.strftime('%Y%m%d')
    return dirname

def create_dir():
    pass


if __name__ == "__main__":
    cli()
