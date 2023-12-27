import pyodbc
import pandas as pd


class SqlConnect:  # Create an object to interact with an SQL server

    def _find_driver(
            self):  # A private function that automatically detects the OBDC driver in a system and selects the most recent version
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

    def __init__(self, server, database=None, driver=None, username=None, password=None,
                 is_trusted=False):  # Initalization of the object
        self.__connection = None  # Initalizing the connection
        self.__server = server  # Server address
        self.__username = username  # Username of the database
        self.__password = password  # Password of the database
        self.__trusted = is_trusted  # If the server is on a local system then longin information would not be required and hence this option must be set to TRUE
        self.database = database  # Name of the database
        self.cursor = None  # Initalizing the cursor

        if driver is None:  # If a specefic driver version is required then this is used to set it
            self.__driver = self._find_driver()
        else:
            self.__driver = driver

    def connect(self):  # Connect to an SQL server

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

    def close(self): #Closes the connection to the SQL server and must be used at some point if connect is used
        if self.__connection:
            self.cursor.close()
            self.__connection.close()

    def insert(self, table_name, values): #Insert data into a table
        # values: A dictionary consisting of the column name and the data to be inserted into it (ie. {name:'Bob',age:24})
        columns = self.get_column_names(table_name)
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]

        organized_values = []
        insert_string = f'INSERT INTO {table_name} ('

        latest_key = self.get_latest_key(table_name)
        if latest_key is None:
            latest_key = 0
        primary_key = self.get_primary_key(table_name)

        for value in values:
            latest_key += 1
            value[primary_key] = latest_key
            for column in columns:
                organized_values.append(value[column])

        for column in columns:
            insert_string = insert_string + column + ","
        insert_string = insert_string[:-1]
        insert_string += ") VALUES"

        for number in range(len(values)):
            placeholder_string = ', '.join(['?' for _ in range(len(columns))])
            insert_string += " (" + placeholder_string + "),"

        insert_string = insert_string[:-1]
        try:
            self.cursor.execute(insert_string, organized_values)
            self.__connection.commit()
        except Exception as e:
            print(e)

    def delete(self, table_name, key): #Delete a row from the table using its primary key
        column_name = self.get_primary_key(table_name)
        delete_string = f"DELETE FROM {table_name} WHERE {column_name} = {key};"
        try:
            self.cursor.execute(delete_string)
            self.__connection.commit()
        except Exception as e:
            print(e)

    def get_data(self, table_name, displayed_columns=None, keys=None, top_rows=None):
        condition_string = ""
        if top_rows is None and displayed_columns is None:
            filter_string = "*"
            displayed_columns = self.get_column_names(table_name)
        else:
            if top_rows:
                top_rows = f"TOP ({top_rows}) "
            else:
                top_rows = ""

            if displayed_columns:
                if not isinstance(displayed_columns, list) and not isinstance(displayed_columns, tuple):
                    displayed_columns = [displayed_columns]
            else:
                displayed_columns = self.get_column_names(table_name)

            coulmn_names_formatted = ", ".join([f'[{column}]' for column in displayed_columns])

            filter_string = top_rows + coulmn_names_formatted

        if keys:
            key_string = ' and '.join([f"{key}={data}" for key, data in keys.items()])
            condition_string = f"WHERE {key_string};"

        get_data_string = f"SELECT {filter_string} FROM {table_name} {condition_string}"

        try:
            self.cursor.execute(get_data_string)
            fetched_data = self.cursor.fetchall()
            returned_data = []
            for data in fetched_data:
                item_data = {}
                for index in range(len(displayed_columns)):
                    item_data[displayed_columns[index]] = data[index]
                returned_data.append(item_data)

            return returned_data

        except Exception as e:
            print(e)

    def get_primary_key(self, table_name):
        table_name = table_name.split(".")[1]
        find_primary_key_string = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + QUOTENAME(CONSTRAINT_NAME)), 'IsPrimaryKey') = 1 AND TABLE_NAME = '{table_name}'"
        try:
            self.cursor.execute(find_primary_key_string)
            primary_key = self.cursor.fetchall()
            if primary_key[0][0]:
                return primary_key[0][0]

        except Exception as e:
            print(e)

    def get_column_names(self, table_name):
        table_name = table_name.split(".")[1]
        get_column_string = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}';"
        try:
            self.cursor.execute(get_column_string)
            true_column_names = []
            for column_name in self.cursor.fetchall():
                true_column_names.append(column_name[0])
            return tuple(true_column_names)

        except Exception as e:
            print(e)

    def get_latest_key(self, table_name):
        primary_key = self.get_primary_key(table_name)
        get_key_string = f"SELECT MAX({primary_key}) FROM {table_name}"
        try:
            self.cursor.execute(get_key_string)
            max_key = self.cursor.fetchall()
            if max_key[0][0]:
                return max_key[0][0]
        except Exception as e:
            print(e)

    def run(self, input_string):
        self.cursor.execute(input_string)
        data = self.cursor.fetchall()
        if data:
            return data