import pandas as pd
import py7zr
from py7zr.io import BytesIOFactory


factory = BytesIOFactory(limit=2 * 1024**3)  # most bytes it will hold in memory (here 2 GB)

with py7zr.SevenZipFile("data.csv.7z", mode="r") as z:
    name = z.getnames()[0]          # or pick the CSV you want by name
    z.extract(targets=[name], factory=factory)

df = pd.read_csv(factory.products[name])