"""Deaths per 100 road accidents in India, 2001 to 2025.

Reads india_road_accidents_2001_2025.csv and saves severity_trend.png.
Run: python severity_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "india_road_accidents_2001_2025.csv")
df["severity"] = df["deaths"] / df["accidents"] * 100  # deaths per 100 accidents

first, last = df.iloc[0], df.iloc[-1]
deaths_per_day = last["deaths"] / 365

BG, INK, MUTED, GREY, RED = "#FBFAF7", "#1F2328", "#6B7178", "#CFD3D8", "#C8452B"
fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
fig.subplots_adjust(left=0.09, right=0.95, top=0.73, bottom=0.17)

highlight = {int(first["year"]), int(last["year"])}
colours = [RED if y in highlight else GREY for y in df["year"]]
ax.bar(df["year"], df["severity"], color=colours, width=0.72)
for row in (first, last):
    ax.text(row["year"], row["severity"] + 0.8, f"{row['severity']:.0f}", ha="center",
            va="bottom", fontsize=20, fontweight="bold", color=RED)

covid = df.loc[df["year"] == 2020, "severity"].iloc[0]
ax.annotate("2020: fewer crashes during\nlockdown, but not fewer deaths\nper crash",
            xy=(2020, covid + 0.3), xytext=(2013.2, 41.5), fontsize=13, color=MUTED,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, connectionstyle="arc3,rad=-0.25"))

ax.set_ylim(0, 45)
ax.set_xlim(df["year"].min() - 0.7, df["year"].max() + 0.7)
ax.set_xticks([2001, 2005, 2010, 2015, 2020, int(last["year"])])
ax.set_yticks([0, 10, 20, 30, 40])
ax.tick_params(colors=MUTED, labelsize=14, length=0, pad=6)
ax.grid(axis="y", color="#E6E3DC", lw=1)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GREY)
ax.set_ylabel("Deaths per 100 road accidents", fontsize=14, color=MUTED, labelpad=10)

fig.text(0.07, 0.93, "Road crashes in India have become far deadlier", fontsize=27, fontweight="bold", color=INK)
fig.text(0.07, 0.875, f"In {int(first['year'])}, about {first['severity']:.0f} people died for every 100 crashes.", fontsize=19, color=INK)
fig.text(0.07, 0.835, f"By {int(last['year'])}, it was {last['severity']:.0f}.", fontsize=19, color=INK)
fig.text(0.07, 0.775, f"That's roughly {round(deaths_per_day, -1):.0f} lives lost every day in {int(last['year'])}.",
         fontsize=17, color=RED, fontweight="bold")
fig.text(0.07, 0.06, "Data: MoRTH, Road Accidents in India reports (2001 to 2024); 2025 figures given to Parliament, July 2026",
         fontsize=12.5, color=MUTED)

fig.savefig(HERE / "severity_trend.png", dpi=100, facecolor=BG)
print(df[["year", "accidents", "deaths", "severity"]].round(1).to_string(index=False))
