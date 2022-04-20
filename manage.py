from datetime import datetime
from pathlib import Path
import shutil

import pytz
import click


@click.group()
def cli():
    pass

@cli.command()
@click.option('--template_type', '-t',
              type=click.Choice(['table', 'timeseries', 'image'], case_sensitive=False))
def startproject(template_type):

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


@cli.command()
@click.option("--project_dir", "-p", type=click.Path(dir_okay=True))
def restartproject(project_dir):


@cli.command()
def collectstatic():
    pass

def get_dirname():
    tz = pytz.timezone('Japan')
    dt = tz.localize(datetime.today())
    dirname = "proj_" + dt.strftime('%Y%m%d')
    return dirname


if __name__ == "__main__":
    cli()
