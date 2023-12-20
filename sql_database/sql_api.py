import pyodbc
import pandas as pd


class SqlConnect:

    def _find_driver(self):
        try:
            driver_list = sorted([driver for driver in pyodbc.drivers() if
                                  driver.startswith('ODBC Driver') and driver.endswith('for SQL Server')])
            if not driver_list:
                raise ValueError(
                    "There are no ODBC drivers in your system. Please install the required drivers or specify its name if already installed")
            else:
                return driver_list[-1]
        except Exception as e:
            print(e)
            raise SystemExit(1)

    def __init__(self, server, database=None, driver=None, username=None, password=None, is_trusted=False):
        self.__connection = None
        self.__server = server
        self.__username = username
        self.__password = password
        self.__trusted = is_trusted
        self.database = database
        self.cursor = None

        if driver is None:
            self.__driver = self._find_driver()
        else:
            self.__driver = driver

    def connect(self):

        if self.__trusted:
            login_string = "TRUSTED_CONNECTION=yes"
        else:
            login_string = f"UID={self.__username};PWD={self.__password}"

        connection_string = f"DRIVER={self.__driver};SERVER={self.__server};DATABASE={self.database};" + login_string

        try:
            self.__connection = pyodbc.connect(connection_string)
            self.cursor = self.__connection.cursor()
        except Exception as e:
            print(e)

    def close(self):
        if self.__connection:
            self.cursor.close()
            self.__connection.close()

    def insert(self, table_name, values, columns):
        organized_values = []
        insert_string = f'INSERT INTO {table_name} ('
        for value in values:
            for column in columns:
                organized_values.append(value[column])

        for column in columns:
            insert_string= insert_string + column +","
        insert_string = insert_string[:-1]
        insert_string+=") VALUES"

        for number in range(len(values)):
            placeholder_string = ', '.join(['?' for _ in range(len(columns))])
            insert_string+=" ("+placeholder_string+"),"

        insert_string = insert_string[:-1]
        try:
            self.cursor.execute(insert_string, organized_values)
            self.__connection.commit()
        except Exception as e:
            print(e)

    def delete(self, table_name, id):
        pass

    def run_script(self, input):
        self.cursor.execute(input)
        data = self.cursor.fetchall()
        if data:
            return data