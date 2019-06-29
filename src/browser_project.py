"""
Created on 20 Jan, 2019

@author: David García Morillo
"""
import csv
import datetime as dt
from collections import namedtuple
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

pd.plotting.register_matplotlib_converters()

plt.style.use("bmh")


def namedtuple_fixed(name: str, fields: List[str]) -> namedtuple:
    """Check the fields of the namedtuple and changes the invalid ones."""

    def starts_with_number(string: str) -> bool:
        if string[0].isdigit():
            return True
        return False

    fields_fixed: List[str] = []
    for field in fields:
        field = field.replace(" ", "_")
        if starts_with_number(field):
            field = f"c{field}"
        fields_fixed.append(field)

    return namedtuple(name, fields_fixed)


Record: namedtuple = namedtuple("Empty_namedtuple", "abc")


def read_file(file: str) -> List["Record"]:
    """
    Read the file with info about the percentage of use of various browsers
    """
    global Record
    with open(file, encoding="UTF-8") as browsers_file:
        reader: Iterator[List[str]] = csv.reader(browsers_file)
        field_names: List[str] = next(reader)
        Record = namedtuple_fixed("Record", field_names)
        result: List[Record] = [
            Record(
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
    data: List[Record],
    initial_date: Optional[str] = None,
    final_date: Optional[str] = None,
) -> List[Record]:
    """
        INPUT = data: list of tuples with the percentage of use
        of certain browsers, on a certain date
        initial_date: Initial date
        final_date: Final date
        OUTPUT = LIst of tuples with the browsers between these dates
    """
    result: List[Record] = data

    if initial_date is not None and final_date is not None:
        result = [
            n
            for n in data
            if dt.datetime.strptime(initial_date, "%Y-%m").date()
            <= n.Date
            <= dt.datetime.strptime(final_date, "%Y-%m").date()
        ]

    elif initial_date is not None:
        result = [
            n
            for n in data
            if n.Date <= dt.datetime.strptime(initial_date, "%Y-%m").date()
        ]

    elif final_date is not None:
        result = [
            n
            for n in data
            if dt.datetime.strptime(final_date, "%Y-%m").date() <= n.Date
        ]

    return result


def filter_by_date_and_browser(
    data: List[Record], date: str, browser: str
) -> Optional[float]:
    """
    Return the percentage of use of a browser in a date given as parameter
    """
    for record in data:
        if dt.datetime.strptime(date, "%Y-%m").date() == record.Date:
            return getattr(record, browser)

    return None


def filter_by_browser(data: List[Record], browser: str) -> List[Tuple[dt.date, float]]:
    """
    Return a list with the percentages of use of a browser 
    given as a parameter along with its corresponding dates 
    """

    return [(record.Date, getattr(record, browser)) for record in data]


def plot_evolution_browsers_between_dates(
    data: List[Record],
    list_of_browsers: List[str],
    *,
    initial_date: Optional[str] = None,
    final_date: Optional[str] = None,
) -> None:
    """
    Plot the evolution of usage of the browsers given as parameter. 
    If a initial or final date is given, plots the percentages between them,
    otherwise plots all the data
    """
    data_filtered: List[Record] = filter_by_date(data, initial_date, final_date)

    percentages: List[List[float]] = []
    dates: List[List[dt.date]] = []

    for browser in list_of_browsers:
        percentages_per_browser: List[float] = []
        dates_per_browser: List[dt.date] = []

        for record in data_filtered:
            percentages_per_browser.append(getattr(record, browser))
            dates_per_browser.append(record.Date)

        percentages.append(percentages_per_browser)
        dates.append(dates_per_browser)

    for i, browser in enumerate(list_of_browsers):
        plt.plot_date(
            dates[i],
            percentages[i],
            label=browser,
            marker="o",
            markerfacecolor="purple",
            markersize=2,
            linestyle="solid",
        )

    plt.legend()
    plt.ylabel("Percentages")
    plt.xlabel("Dates")
    plt.title(
        (
            f"Evolution of browser usage between {dt.datetime.strftime(dates[0][0], '%b-%Y')} "
            f"and {dt.datetime.strftime(dates[0][-1], '%b-%Y')}"
        )
    )

    fig: plt.Figure = plt.gcf()
    fig.autofmt_xdate()

    plt.margins(x=0, y=0)

    plt.tight_layout()
    plt.show()


def plot_stick_graph(data: List[Record], list_of_browsers: List[str]) -> None:
    """
    Plot a bar chart with the means of usage 
    of the browsers given as parameter
    """
    means_of_usage: List[float] = [
        np.mean([getattr(record, browser) for record in data])
        for browser in list_of_browsers
    ]

    sns.barplot(means_of_usage, list_of_browsers, palette="rocket", orient="h")

    plt.title("Average use of browsers")

    plt.tight_layout()
    plt.show()


def filter_dataframe_by_list_of_browsers(
    file: str, list_of_browsers: List[str]
) -> pd.DataFrame:
    """
    Return a pandas.DataFrame with the 
    percentage of use of the browsers given as parameter
    """

    data_frame: pd.DataFrame = pd.read_csv(
        file, index_col="Date", usecols=list_of_browsers + ["Date"]
    )

    outside: List[str] = []
    Record: List[float] = []

    for date in list(data_frame.index):
        outside.extend([date] * len(data_frame.columns))
        Record.extend(list(data_frame.loc[date]))

    inside: List[str] = list(data_frame.columns) * len(data_frame)

    assert len(inside) == len(outside)

    hier_index: pd.MultiIndex = pd.MultiIndex.from_tuples(
        zip(outside, inside), names=["Dates", "Browsers"]
    )

    data_frame_with_indexes: pd.DataFrame = pd.DataFrame(
        data=Record, index=hier_index, columns=["Record"]
    )

    return data_frame_with_indexes


def filter_dataframe_by_importance(file: str, filter_: float = 0.0) -> pd.DataFrame:
    """
    Return a pandas.DataFrame with the browsers which mean 
    is greater than the filter given as parameter
    """
    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")
    data_frame_transposed: pd.DataFrame = data_frame.transpose()
    return data_frame_transposed[data_frame.mean() > filter_]


def plot_evolution_browsers_between_dates_with_data_frame(
    file: str,
    list_of_browsers: List[str],
    *,
    initial_date: Optional[str] = None,
    final_date: Optional[str] = None,
) -> None:
    """
    Plot a chart with the evolution of usage of the browsers given as parameter.
    """

    data_frame: pd.DataFrame = pd.read_csv(
        file, index_col="Date", usecols=list_of_browsers + ["Date"], parse_dates=True
    )

    if initial_date is not None and final_date is not None:
        _initial_date: dt.date = dt.datetime.strptime(initial_date, "%Y-%m")
        _final_date: dt.date = dt.datetime.strptime(final_date, "%Y-%m")

        data_frame = data_frame[
            (_initial_date <= data_frame.index) & (data_frame.index <= _final_date)
        ]

    elif initial_date is not None:
        _initial_date = dt.datetime.strptime(initial_date, "%Y-%m")

        data_frame = data_frame[_initial_date <= data_frame.index]

    elif final_date is not None:
        _final_date = dt.datetime.strptime(final_date, "%Y-%m")

        data_frame = data_frame[data_frame.index <= _final_date]

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

    plt.title(
        (
            f"Evolution of browser usage between {dt.datetime.strftime(data_frame.index[0], '%b, %Y')} "
            f"and {dt.datetime.strftime(data_frame.index[-1], '%b, %Y')}"
        )
    )
    plt.xlabel("Dates")
    plt.ylabel("Percentage of use")

    plt.tight_layout()
    plt.show()


def filter_by_date_and_browser_with_dataframes(
    file: str, date: str, browser: str
) -> float:
    """
    Return the percentage of use of a browser on a certain date.
    """
    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")

    return data_frame[browser].loc[date]


def plot_pie_chart(
    file: str, date: Optional[str] = None, *, circle: bool = False
) -> None:
    """
    Plot a simple pie chart.
    If circle parameter is set to True, the chart is converted to a circle object
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
