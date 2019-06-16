"""
Created on 20 ene. 2019

@author: David
"""
from datetime import datetime as dt
from pprint import pprint
from typing import List, Tuple

import pandas as pd

import browser_project


def test_filter_by_date(data) -> None:
    init_date = "2017-09"
    final_date = "2018-02"
    result: browser_project.Records = browser_project.filter_by_date(
        data, init_date, final_date
    )
    if init_date is None is final_date:
        print("Printing all records:")
    elif final_date is None:
        print(f"Printing records since {init_date}:")
    elif init_date is None:
        print(f"Printing records before {final_date}:")
    else:
        print(f"Browser records between {init_date} and {final_date}:")
    pprint(result)
    print()


def test_filter_by_date_and_browser(data) -> None:
    date = "2018-07"
    browser = "Safari"
    percentage: float = browser_project.filter_by_date_and_browser(data, date, browser)
    print(
        f"The percentage of usage of the browser {browser} in {date} is {percentage}\n"
    )


def test_filter_by_browser(data) -> None:
    browser = "Chrome"
    result: List[Tuple[dt.date, float]] = browser_project.filter_by_browser(
        data, browser
    )
    print(f"Printing the percentages of use of the browser {browser}:")
    pprint(result)
    print()


def test_filter_data_frame_by_list_of_browsers(
    file: str, list_of_browser: list, display_all: bool = False
) -> None:
    data_frame: pd.DataFrame = browser_project.filter_dataframe_by_list_of_browsers(
        file, list_of_browser
    )
    if display_all:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(data_frame, "\n")
    else:
        print(data_frame, "\n")


def test_filter_data_frame_by_importance(data, filter_) -> None:
    data_frame: pd.DataFrame = browser_project.filter_dataframe_by_importance(
        data, filter_
    )
    print(
        f"Printing the DataFrame with the browsers which mean is greater than {filter_}"
    )
    print(data_frame)


def test_filter_by_date_and_browser_with_data_frame(file, date, browser) -> None:
    print(f"The percentage of use of the browser {browser} on the date {date} is:")
    print(
        browser_project.filter_by_date_and_browser_with_dataframes(file, date, browser)
    )


if __name__ == "__main__":
    # ===========================================================================
    FILE_2009_to_2019 = "./data/browser-ww-monthly-200901-201902.csv"
    FILE_2017_to_2018 = "./data/browser-ww-monthly-201701-201812.csv"

    records: List[browser_project.Records] = browser_project.read_file(
        FILE_2017_to_2018
    )

    pprint(records[:5])
    print()

    test_filter_by_date(records)

    test_filter_by_date_and_browser(records)

    test_filter_by_browser(records)

    browser_project.plot_graph_browser(
        records, ["Chrome", "Firefox", "Edge", "Safari", "IE"]
    )

    browser_project.plot_stick_graph(
        records, ["Chrome", "Firefox", "Edge", "Safari", "IE"]
    )

    test_filter_data_frame_by_list_of_browsers(
        FILE_2009_to_2019, ["Chrome", "Firefox", "Edge", "Safari", "IE"]
    )

    test_filter_data_frame_by_importance(FILE_2009_to_2019, 5.0)

    browser_project.plot_evolution_browsers(
        FILE_2009_to_2019, ["Chrome", "Firefox", "Edge", "Safari", "IE"]
    )

    test_filter_by_date_and_browser_with_data_frame(
        FILE_2017_to_2018, "2017-02", "Firefox"
    )

    browser_project.plot_pie_chart(FILE_2017_to_2018, "2017-02", circle=True)
    # ===========================================================================
