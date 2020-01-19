"""
Created on 20 Jan, 2019

@author: David García Morillo
"""
import csv
import datetime as dt
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Type

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

pd.plotting.register_matplotlib_converters()

plt.style.use("bmh")


def namedtuple_fixed(name: str, fields: List[str]) -> NamedTuple:
    """Check the fields of the namedtuple and changes the invalid ones."""

    fields_fixed: Dict[str, type] = {}

    for field in fields:
        field = field.replace(" ", "_")

        if field[0].isdigit():
            field = f"n{field}"

        fields_fixed[field] = float
    
    fields_fixed["Date"] = dt.date
    


    return NamedTuple(name, **fields_fixed)


Record: NamedTuple = NamedTuple("Empty", a=int)
# Empty namedtuple. When 'read_file' is executed,
# turns to a namedtuple fixed to the file containing the browsers' names

# pylint: disable=used-prior-global-declaration, not-callable
def read_file(file: str) -> "List[Record]":
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
    Return the data between the dates given as parameter.
    If no initial or final date given, returns all data
    """
    result: List[Record] = data

    if initial_date is not None and final_date is not None:
        result = [
            record
            for record in data
            if dt.datetime.strptime(initial_date, "%Y-%m").date()
            <= record.Date
            <= dt.datetime.strptime(final_date, "%Y-%m").date()
        ]

    elif initial_date is not None:
        result = [
            record
            for record in data
            if record.Date <= dt.datetime.strptime(initial_date, "%Y-%m").date()
        ]

    elif final_date is not None:
        result = [
            record
            for record in data
            if dt.datetime.strptime(final_date, "%Y-%m").date() <= record.Date
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
            f"Evolution of browser usage between {dates[0][0]: '%b-%Y'} "
            f"and {dates[0][-1]: '%b-%Y'}"
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


def dataframe_browsers_grouped_hierarchically(
    file: str, list_of_browsers: Optional[List[str]] = None, filter_: float = 0.0
) -> pd.DataFrame:
    """
    Return a pandas.DataFrame with the 
    percentage of use of the browsers of the csv file
    given as parameter, the pandas.DataFrame is grouped
    hierarchically by date and browsers.

    Parameters
    ----------
    list_of_browsers: If given, filter the pandas.DataFrame with them.
    filter_: If given, filter the pandas.DataFrame with the browsers which mean is greater than it 
    
    Cannot specify both list_of_browsers and filter_
    """
    if filter_ and list_of_browsers:
        raise ValueError("Cannot specify both list_of_browsers and filter_")

    data_frame: pd.DataFrame = pd.read_csv(
        file,
        usecols=(list_of_browsers + ["Date"]) if list_of_browsers else None,
        index_col="Date",
    )

    data_frame = data_frame[data_frame.columns[data_frame.mean() > filter_]]

    data_frame_columns: pd.Index = data_frame.columns

    outside: List[str] = []
    data: List[float] = []

    for date in data_frame.index:
        outside.extend([date] * len(data_frame_columns))
        data.extend(data_frame.loc[date])

    inside: List[str] = data_frame_columns.to_list() * len(data_frame)

    assert len(inside) == len(outside)

    hier_index: pd.MultiIndex = pd.MultiIndex.from_tuples(
        zip(outside, inside), names=["Dates", "Browsers"]
    )

    data_frame_with_indexes: pd.DataFrame = pd.DataFrame(
        data=data, index=hier_index, columns=["Records"]
    )

    return data_frame_with_indexes


def dataframe_browsers(
    file: str, list_of_browsers: Optional[List[str]] = None, filter_: float = 0.0
) -> pd.DataFrame:
    """
    Return a pandas.DataFrame with the 
    percentage of use of the browsers of the csv file given as parameter.
    
    Parameters
    ----------
    list_of_browsers: If given, filter the pandas.DataFrame with them.
    filter_: If given, filter the pandas.DataFrame with the browsers which mean is greater than it

    (The difference between dataframe_browsers and this function is that the first function 
    group the percentages by the dates and browsers)
    """
    if filter_ and list_of_browsers:
        raise ValueError("Cannot specify both list_of_browsers and filter_")

    data_frame: pd.DataFrame = pd.read_csv(
        file,
        index_col="Date",
        usecols=(list_of_browsers + ["Date"]) if list_of_browsers else None,
    )
    data_frame_transposed: pd.DataFrame = data_frame.transpose()

    return data_frame_transposed[data_frame.mean() > filter_]


def plot_evolution_browsers_use_between_dates_with_data_frame(
    file: str,
    list_of_browsers: List[str],
    *,
    initial_date: Optional[str] = None,
    final_date: Optional[str] = None,
    fill_between: bool = False,
) -> None:
    """
    Plot a chart with the evolution of usage of the browsers given as parameter.
    """

    data_frame: pd.DataFrame = pd.read_csv(
        file, index_col="Date", usecols=list_of_browsers + ["Date"], parse_dates=True
    )

    if initial_date is not None and final_date is not None:
        initial_date_as_datetime: dt.datetime = dt.datetime.strptime(
            initial_date, "%Y-%m"
        )
        final_date_as_datetime: dt.datetime = dt.datetime.strptime(final_date, "%Y-%m")

        data_frame = data_frame[
            (initial_date_as_datetime <= data_frame.index)
            & (data_frame.index <= final_date_as_datetime)
        ]

    elif initial_date is not None:
        initial_date_as_datetime = dt.datetime.strptime(initial_date, "%Y-%m")

        data_frame = data_frame[initial_date_as_datetime <= data_frame.index]

    elif final_date is not None:
        final_date_as_datetime = dt.datetime.strptime(final_date, "%Y-%m")

        data_frame = data_frame[data_frame.index <= final_date_as_datetime]

    sns.lineplot(data=data_frame, palette="bright")

    if fill_between:
        for browser in data_frame:

            plt.fill_between(
                data_frame.index, data_frame[browser], alpha=0.3, interpolate=True
            )

    plt.title(
        (
            f"Evolution of browser usage between {data_frame.index[0]: '%b, %Y'}"
            f" and {data_frame.index[-1]: '%b, %Y'}"
        )
    )
    plt.xlabel("Dates")
    plt.ylabel("Percentage of use")

    plt.margins(x=0, y=0)

    plt.tight_layout()
    plt.show()


def filter_by_date_and_browser_with_dataframes(
    file: str, date: str, browser: str
) -> float:
    """
    Return the percentage of use of a browser on a certain date.
    """
    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")

    return data_frame.loc[date, browser]


def plot_pie_chart(
    file: str, date: Optional[str] = None, *, circle: bool = False
) -> None:
    """
    Plot a simple pie chart with the means of the browsers usage,

    Parameters
    ----------

    file: String, CSV file containing the browsers use percentage
    date: String, if given, plot the registers of that date.
    circle: Boolean, if set to True, the chart is converted to a circle object, default is False
    """
    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")

    data: pd.Series = data_frame.loc[date] if date else data_frame.mean()
    data = data[data > 1.0]  # Get only the browsers with significant percentage

    data["Other"] = 100 - sum(data)
    # Group the remaining percentage in another column

    data.sort_values(ascending=False, inplace=True)

    explodes: Tuple[float, ...] = (0.1,) + (0.0,) * (len(data) - 1)
    plt.pie(
        data,
        labels=data.index,
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


def statistics_metrics_by_browsers(
    file: str,
    list_of_browsers: Optional[List[str]] = None,
    *,
    filter_by: Optional[Tuple[str, float]] = None,
    sort_by_statistic_func: str = "mean",
    transpose: bool = False,
) -> pd.DataFrame:
    """
    Return a pandas.DataFrame consinsting of statistics metrics.

    Parameters
    ----------

    file: CSV file with the percentages of use\n

    list_of_browsers: List of str, if given, filter the pandas.DataFrame with the browsers in it\n

    filter_by: Tuple of a str and a float, first element refere to the statistic
    function that must be greater than the second one. If given, filters by the browsers
    which accomplish the condition mentioned before\n

    sort_by: String, it should refere to a statistic function. If given,
    sort the pandas.DataFrame by it. Default is 'mean'\n

    transpose: Boolean, if set to True, returns the pandas.DataFrame transposed. Default is False.
    """

    if list_of_browsers and filter_by:
        raise ValueError("Cannot specify both list_of_browsers and filter_by")

    data_frame: pd.DataFrame = dataframe_browsers_grouped_hierarchically(
        file, list_of_browsers
    ).groupby("Browsers").describe().sort_values(
        by=("Records", sort_by_statistic_func), ascending=False
    )

    if filter_by:
        data_frame = data_frame[data_frame[("Records", filter_by[0])] > filter_by[1]]

    if transpose:
        return data_frame.transpose().round(2)

    return data_frame


def plot_box_chart(
    file: str, list_of_browsers: Optional[List[str]] = None, limit: Optional[int] = None
) -> None:
    """
    Plot box chart

    Parameters
    ----------
    file: CSV file with the percentages of use.\n
    
    list_of_browsers: List of strings, if given, 
    filter the pandas.DataFrame with the browsers in it.\n
    
    limit: Integer, if given, plots only the boxplot of the first n browsers.
    """
    if list_of_browsers and limit:
        raise ValueError("Cannot specify both list_of_browsers and limit")

    data_frame: pd.DataFrame = pd.read_csv(file, index_col="Date")

    if list_of_browsers:
        sns.boxplot(data=data_frame[list_of_browsers])

    elif limit:
        sns.boxplot(data=data_frame.iloc[:, :limit])

    else:
        sns.boxplot(data=data_frame)

    plt.show()
