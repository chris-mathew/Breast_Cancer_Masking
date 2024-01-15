import os
from sql_database.sql_api import SqlConnect
from io import BytesIO
from PIL import Image

#A sub class of the SQLConnect class that is specefic to the DDSM dataset
class DDSMDataset(SqlConnect):

    def __init__(self):
        self.table_name = "dbo.ddsm_dataset"
        super().__init__(server="ctrl-alt-elite.database.windows.net", database="ai_brestcancer",
                         username="ctrl-alt-elite", password="Tsnte7TF6nMZTPY")
#Insert data as a list with column names
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

                if image_name_split[5] == "BENIGN":
                    folder_value['cancer'] = 0
                elif image_name_split[5] == "MALIGNANT":
                    folder_value['cancer'] = 1

                super().insert(self.table_name, [folder_value])
                print(f"{folder} has been uploaded")

        super().close()
#Get the latest group id
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
#Get an individual patient's data that includes images of all four view. If single is set to true then it would only return an individual image
    def get_grouped_data(self, index, single=False):
        index += 1
        super().connect()
        if single:
            key = {"id": index}
        else:
            key = {"group_id": index}
        data = super().get_data("dbo.ddsm_dataset", keys=key)
        super().close()
        for item in range(len(data)):
            #Converting BIT values to their string equivalent
            if data[item]['direction']:
                data[item]['direction'] = 'RIGHT'
            else:
                data[item]['direction'] = 'LEFT'
            if data[item]['image_view']:
                data[item]['image_view'] = 'MLO'
            else:
                data[item]['image_view'] = 'CC'

            if data[item]['cancer']:
                data[item]['cancer'] = "MALIGNANT"
            else:
                data[item]['cancer'] = "BENIGN"
            
            #Image output is converted from string to Bytes
            data[item]['pixel_data'] = Image.open(BytesIO(data[item]['pixel_data'])).convert('RGB')

        if len(data) == 1:
            data = data[0]
        return data
#View an image on the database using PIL
    def view_image(self, index):
        data = self.get_grouped_data(index)

        for item in data:
            image = Image.open(BytesIO(item['pixel_data']))
            image.show()
#Get the number of patiens
    def get_length(self):
        self.connect()
        length = self._get_groupid(self.table_name)
        self.close()
        return int(length)