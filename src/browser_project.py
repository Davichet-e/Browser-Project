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
    Read the file with info about the percentage of use of various browsers
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
    data: List[Records],
    initial_date: Optional[str] = None,
    final_date: Optional[str] = None,
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
    Returns the percentage of use of a browser in a date given as parameter
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


def plot_evolution_browsers_by_dates(
    data: List[Records],
    list_of_browsers: List[str],
    *,
    init_date: Optional[str] = None,
    final_date: Optional[str] = None,
) -> None:
    """
    Plots the evolution of usage of the browsers given as parameter. 
    If a initial or final date is given, plots the percentages between them,
    otherwise plots all the data
    """
    data_filtered: List[Records] = filter_by_date(data, init_date, final_date)

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
    plt.title("Percentage of use")

    fig = plt.gcf()
    fig.autofmt_xdate()

    plt.margins(x=0, y=0)

    plt.tight_layout()
    plt.show()


def plot_stick_graph(data: List[Records], list_of_browsers: List[str]) -> None:
    """
    Plots a bar chart with the means of usage of the browsers given as parameter
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
    Returns a pandas.DataFrame with the percentage of use of the browsers given as parameter
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
    Returns a pandas.DataFrame with the browsers which mean is greater than the filter given as parameter
    """
    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")
    data_frame_transposed: pd.DataFrame = data_frame.transpose()
    return data_frame_transposed[data_frame.mean() > filter_]


def plot_evolution_browsers_by_dates_with_data_frame(
    file: str,
    list_of_browsers: List[str],
    *,
    init_date: Optional[str] = None,
    final_date: Optional[str] = None,
) -> None:
    """
    Plots a chart with the evolution of usage of the browsers given as parameter
    """
    list_of_browsers.append("Date")

    data_frame: pd.DataFrame = pd.read_csv(
        file, index_col="Date", usecols=list_of_browsers, parse_dates=True
    )

    if init_date is not None and final_date is not None:
        init_date: dt.date = dt.datetime.strptime(init_date, "%Y-%m")
        final_date: dt.date = dt.datetime.strptime(final_date, "%Y-%m")

        data_frame = data_frame[
            (init_date <= data_frame.index) & (data_frame.index <= final_date)
        ]

    elif init_date is not None:
        init_date: dt.date = dt.datetime.strptime(init_date, "%Y-%m")

        data_frame = data_frame[init_date <= data_frame.index]

    elif final_date is not None:
        final_date: dt.date = dt.datetime.strptime(final_date, "%Y-%m")

        data_frame = data_frame[data_frame.index <= final_date]

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
    """Returns the percentage of use of a browser on a certain date"""
    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")

    return data_frame[browser].loc[date]


def plot_pie_chart(
    file: str, date: Optional[str] = None, *, circle: bool = False
) -> None:
    """
    Plots a simple pie chart.
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
