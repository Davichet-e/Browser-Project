"""
Created on 20 ene. 2019

@author: David García Morillo
"""
import csv
import datetime as dt
from collections import namedtuple
from statistics import mean
from typing import List, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

pd.plotting.register_matplotlib_converters()

plt.style.use("bmh")


def namedtuple_fixed(name: str, fields: List[str]) -> namedtuple:
    """Checks the fields of the namedtuple and changes the invalid ones."""

    def starts_with_number(string: str) -> bool:
        if string[0].isdigit():
            return True
        return False

    fields_fixed: List[str] = [
        field.replace(" ", "_")
        if not starts_with_number(field)
        else f"c{field.replace(' ', '_')}"
        for field in fields
    ]

    return namedtuple(name, fields_fixed)


Records: namedtuple = None


def read_file(file: str) -> List["Records"]:
    """
    Read the file with info about the percentage of usage of the browsers around the world.
        INPUT = The file with the data
        OUTPUT = List of tuples with the info of the csv file
    """
    global Records

    with open(file, encoding="UTF-8") as browsers_file:
        reader = csv.reader(browsers_file)
        field_names = next(reader)
        Records = namedtuple_fixed("Record", field_names)
        result: List[Records] = [
            Records(
                *[
                    dt.datetime.strptime(n, "%Y-%m").date()
                    if record.index(n) == 0
                    else float(n)
                    for n in record
                ]
            )
            for record in reader
        ]
    return result


def filter_by_date(
    data: List[Records], initial_date: str, final_date: str
) -> List[Records]:
    """
        INPUT = data: list of tuples with the percentage of use
        of certain browsers, on a certain date
        initial_date: Initial date
        final_date: Final date
        OUTPUT = LIst of tuples with the browsers between these dates
    """
    result: List[Records]

    if initial_date is None is final_date:
        result = data

    elif initial_date is None:
        result = [
            n
            for n in data
            if n.Date <= dt.datetime.strptime(final_date, "%Y-%m").date()
        ]

    elif final_date is None:
        result = [
            n
            for n in data
            if dt.datetime.strptime(initial_date, "%Y-%m").date() <= n.Date
        ]

    else:
        result = [
            n
            for n in data
            if dt.datetime.strptime(initial_date, "%Y-%m").date()
            <= n.Date
            <= dt.datetime.strptime(final_date, "%Y-%m").date()
        ]

    return result


def filter_by_date_and_browser(
    data: List[Records], date: str, browser: str
) -> Optional[float]:
    """
    INPUT =
    OUTPUT =
    """
    for record in data:
        if dt.datetime.strptime(date, "%Y-%m").date() == record.Date:
            return getattr(record, browser)

    return None


def filter_by_browser(data: List[Records], browser: str) -> List[Tuple[dt.date, float]]:
    """
    INPUT = data: list of tuples with the percentage of use
    of certain browsers, on a certain date
    Browser: the name of a browser
    OUTPUT = a list of tuples with the percentage of use of the browser and the dates
    """

    return [(record.Date, getattr(record, browser)) for record in data]


def plot_graph_browser(data: List[Records], list_of_browsers: List[str]) -> None:
    """
    INPUT =
    OUTPUT =
    """
    records_by_browser: List[List[Tuple[dt.date, float]]] = [
        filter_by_browser(data, browser) for browser in list_of_browsers
    ]
    percentages: List[List[float]] = [
        [n[1] for n in record] for record in records_by_browser
    ]
    dates: List[List[dt.date]] = [
        [n[0] for n in record] for record in records_by_browser
    ]

    for i, browser in enumerate(list_of_browsers):
        plt.plot(
            dates[i],
            percentages[i],
            label=browser,
            marker="o",
            markerfacecolor="purple",
            markersize=2,
        )
    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Percentages")
    plt.xlabel("Dates")
    plt.title("Percentage of use")

    plt.tight_layout()
    plt.show()


def plot_stick_graph(data: List[Records], list_of_browsers: List[str]) -> None:
    """
    INPUT =
    OUTPUT =
    """
    means_of_usage: List[float] = [
        mean([getattr(record, browser) for record in data])
        for browser in list_of_browsers
    ]

    sns.barplot(means_of_usage, list_of_browsers, palette="rocket", orient="h")
    plt.tight_layout()
    plt.show()


def filter_dataframe_by_list_of_browsers(
    file: str, list_of_browsers: List[str]
) -> pd.DataFrame:
    """
    INPUT =
    OUTPUT =
    """
    list_of_browsers.append("Date")

    data_frame: pd.DataFrame = pd.read_csv(
        file, index_col="Date", usecols=list_of_browsers
    )

    outside: List[str] = []
    records: List[float] = []

    for date in list(data_frame.index):
        outside.extend([date] * len(data_frame.columns))
        records.extend(list(data_frame.loc[date]))

    inside: List[str] = list(data_frame.columns) * len(data_frame)

    assert len(inside) == len(outside)

    hier_index: pd.MultiIndex = pd.MultiIndex.from_tuples(
        zip(outside, inside), names=["Dates", "Browsers"]
    )

    data_frame_with_indexes: pd.DataFrame = pd.DataFrame(
        data=records, index=hier_index, columns=["Records"]
    )

    return data_frame_with_indexes


def filter_dataframe_by_importance(file: str, filter_: float = 0.0) -> pd.DataFrame:
    """
    INPUT =
    OUTPUT =
    """
    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")
    data_frame_filtered = data_frame[data_frame.columns[data_frame.mean() > filter_]]
    return data_frame_filtered.transpose()


def plot_evolution_browsers(file: str, list_of_browsers: List[str]) -> None:
    """
    TODO
    """
    list_of_browsers.append("Date")

    data_frame: pd.DataFrame = pd.read_csv(
        file, index_col="Date", usecols=list_of_browsers, parse_dates=True
    )

    data_frame.plot(
        legend=True,
        grid=True,
        colormap="inferno",
        title="Evolution of browser/s usage",
        marker="o",
        markerfacecolor="red",
        markersize=2,
    )
    for browser in data_frame:

        plt.fill_between(
            data_frame.index, data_frame[browser], alpha=0.3, interpolate=True
        )

    plt.xlabel("Dates")
    plt.ylabel("Percentage of use")

    plt.tight_layout()
    plt.show()


def filter_by_date_and_browser_with_dataframes(
    file: str, date: str, browser: str
) -> float:
    """
    TODO
    """
    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")

    return data_frame[browser].loc[date]


def plot_pie_chart(
    file: str, date: Optional[str] = None, *, circle: bool = False
) -> None:
    """
    TODO
    """
    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")

    data: pd.Series = data_frame.mean() if date is None else data_frame.loc[date]
    data_filtered: pd.Series = data[data > 1.0]

    data_filtered["Other"] = 100 - sum(data_filtered)
    
    data_filtered.sort_values(ascending=False, inplace=True)

    explodes: Tuple[float, ...] = (0.1,) + (0.0,) * (len(data_filtered) - 1)
    plt.pie(
        data_filtered,
        labels=data_filtered.index,
        explode=explodes,
        shadow=True,
        autopct="%1.2f%%",
        pctdistance=0.85,
    )

    if circle:
        centre_circle: plt.Circle = plt.Circle((0, 0), 0.70, fc="white")
        axes: Axes = plt.gca()

        axes.add_artist(centre_circle)

    plt.tight_layout()
    plt.show()
