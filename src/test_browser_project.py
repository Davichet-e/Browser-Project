"""
Created on 20 Jan 2019

@author: David García Morillo
"""
import datetime as dt
from pprint import pprint
from typing import List, Tuple, Optional

import pandas as pd

import browser_project
from browser_project import Record


def test_filter_by_date(
    data: List[Record],
    initial_date: Optional[str] = None,
    final_date: Optional[str] = None,
) -> None:

    result: List[Record] = browser_project.filter_by_date(
        data, initial_date, final_date
    )
    if initial_date is None is final_date:
        print("Printing all records:")
    elif final_date is None:
        print(f"Printing records since {initial_date}:")
    elif initial_date is None:
        print(f"Printing records before {final_date}:")
    else:
        print(f"Browser records between {initial_date} and {final_date}:")
    pprint(result)
    print()


def test_filter_by_date_and_browser(
    data: List[Record], date: str, browser: str
) -> None:

    percentage: Optional[float] = browser_project.filter_by_date_and_browser(
        data, date, browser
    )
    print(
        f"The percentage of usage of the browser {browser} in {date} is {percentage}\n"
    )


def test_filter_by_browser(data: List[Record], browser: str) -> None:

    result: List[Tuple[dt.date, float]] = browser_project.filter_by_browser(
        data, browser
    )
    print(f"Printing the percentages of use of the browser {browser}:")
    pprint(result)
    print()


def test_dataframe_browsers(
    file: str,
    *,
    type_: int,
    list_of_browsers: Optional[List[str]] = None,
    filter_: float = 0.0,
    display_all: bool = False,
) -> None:
    """
    Since both browser_project.dataframe_browsers
    and browser_project.dataframe_browsers2 are pretty similar,
    if type_ is 1, returns the first function, otherwise, the number 2
    """
    data_frame: pd.DataFrame

    if type_ == 1:
        data_frame = browser_project.dataframe_browsers_grouped_hierarchically(
            file, list_of_browsers, filter_
        )
    else:
        data_frame = browser_project.dataframe_browsers(file, list_of_browsers, filter_)

    if filter_:
        print(
            f"Printing the DataFrame with the browsers which mean is greater than {filter_}\n"
        )
    else:
        print("Printing the DataFrame:\n")

    if display_all:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(data_frame, "\n")
    else:
        print(data_frame, "\n")


def test_filter_by_date_and_browser_with_data_frame(
    file: str, date: str, browser: str
) -> None:

    print(f"The percentage of use of the browser {browser} on the date {date} is:")
    print(
        browser_project.filter_by_date_and_browser_with_dataframes(file, date, browser),
        "\n",
    )


def test_statistics_metrics_by_browsers(
    file: str,
    list_of_browsers: Optional[List[str]] = None,
    *,
    filter_by: Optional[Tuple[str, float]] = None,
    sort_by: str = "mean",
    transpose: bool = False,
) -> None:

    print(
        browser_project.statistics_metrics_by_browsers(
            file,
            list_of_browsers,
            filter_by=filter_by,
            sort_by=sort_by,
            transpose=transpose,
        )
    )


if __name__ == "__main__":
    # ===========================================================================
    FILE_2009_TO_2019 = "./data/browser-ww-monthly-200901-201902.csv"
    FILE_2017_TO_2018 = "./data/browser-ww-monthly-201701-201812.csv"

    records: List[Record] = browser_project.read_file(FILE_2009_TO_2019)

    pprint(records[:5])
    print()

    test_filter_by_date(records, initial_date="2016-11")

    test_filter_by_date_and_browser(records, "2015-07", "Safari")

    test_filter_by_browser(records, "IE")

    browser_project.plot_evolution_browsers_between_dates(
        records, ["Chrome", "Firefox", "Edge", "Safari", "IE"], initial_date="2013-05"
    )

    browser_project.plot_stick_graph(
        records, ["Chrome", "Firefox", "Edge", "Safari", "IE"]
    )

    test_dataframe_browsers(FILE_2009_TO_2019, type_=2, filter_=5.0)

    browser_project.plot_evolution_browsers_use_between_dates_with_data_frame(
        FILE_2009_TO_2019, ["Chrome", "Firefox", "Edge", "Safari", "IE"]
    )

    test_filter_by_date_and_browser_with_data_frame(
        FILE_2017_TO_2018, "2017-02", "Firefox"
    )

    browser_project.plot_pie_chart(FILE_2017_TO_2018, "2017-02", circle=True)

    test_statistics_metrics_by_browsers(FILE_2017_TO_2018)

    browser_project.plot_box_chart(FILE_2009_TO_2019, limit=4)
    # ===========================================================================
