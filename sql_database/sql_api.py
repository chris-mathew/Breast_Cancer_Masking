import pyodbc
import pandas as pd
import os

###################################################
#### Ensure the connection is closed when done ####
###################################################

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

    def get_data(self, table_name, displayed_columns=None, keys=None, top_rows=None): #Get data from the SQL server
        #displayed_columns: Input a list of the columns that you would like to get data from. This is to avoid loading unwanted data
        #keys: A dictionary of coulmn names with their values that could set a criteria of the data you want to return. For example input {age:24} to reciveve data from people with the age of 24.
        #top_rows: A string of the number of data you would want to return. For example if a table has a million rows and you only want the first 20 then set this variable to 20
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

    def get_primary_key(self, table_name): #Get the primary key of the table which is what the table uses to identify each row (Similar to a row number)
        table_name = table_name.split(".")[1]
        find_primary_key_string = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + QUOTENAME(CONSTRAINT_NAME)), 'IsPrimaryKey') = 1 AND TABLE_NAME = '{table_name}'"
        try:
            self.cursor.execute(find_primary_key_string)
            primary_key = self.cursor.fetchall()
            if primary_key[0][0]:
                return primary_key[0][0]

        except Exception as e:
            print(e)

    def get_column_names(self, table_name): #Get the names of all the columns in a table
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

    def get_latest_key(self, table_name): #Get the biggest value of a column using the int data type. This is useful to identify the last primary key value
        primary_key = self.get_primary_key(table_name)
        get_key_string = f"SELECT MAX({primary_key}) FROM {table_name}"
        try:
            self.cursor.execute(get_key_string)
            max_key = self.cursor.fetchall()
            if max_key[0][0]:
                return max_key[0][0]
        except Exception as e:
            print(e)

    def run(self, input_string,commit=False): #Run a custom SQL command
        self.cursor.execute(input_string)
        if commit:
            self.cursor.commit()
        data = self.cursor.fetchall()
        if data:
            return data


class DDSMDataset(SqlConnect):

    def __init__(self):
        self.table_name = "dbo.ddsm_dataset"
        super().__init__(server="ctrl-alt-elite.database.windows.net", database="ai_brestcancer", username="ctrl-alt-elite", password="Tsnte7TF6nMZTPY")

    def insert_data(self, path):
        folder_names = os.listdir(path)
        super().connect()
        max_group_id = self._get_groupid(self.table_name)
        if max_group_id is None:
            max_group_id = 0
        for dir in folder_names:
            max_group_id += 1
            dirnames = os.listdir(path + "/" + dir)
            for folder in dirnames:
                folder_value = {}
                folder_value['group_id'] = max_group_id
                with open(path + "/" + dir + '/' + folder, 'rb') as file:
                    folder_value['pixel_data'] = file.read()
                image_name_split = folder.split(".")[0].split("_")
                folder_value['patient_id'] = int(image_name_split[1])
                if image_name_split[2] == 'LEFT':
                    folder_value['direction'] = 0
                else:
                    folder_value['direction'] = 1
                if image_name_split[3] == "CC":
                    folder_value['image_view'] = 0
                else:
                    folder_value['image_view'] = 1
                folder_value['density'] = int(image_name_split[4])

                super().insert(self.table_name, [folder_value])
                print(f"{folder} has been uploaded")

        super().close()

    def _get_groupid(self, table_name):
        column_name = "group_id"
        get_key_string = f"SELECT MAX({column_name}) FROM {table_name}"
        try:
            self.cursor.execute(get_key_string)
            max_key = self.cursor.fetchall()
            if max_key[0][0]:
                return max_key[0][0]
        except Exception as e:
            print(e)

    def get_grouped_data(self, index):
        super().connect()
        key = {"group_id": index}
        data = super().get_data("dbo.ddsm_dataset", keys=key)
        super().close()
        return data