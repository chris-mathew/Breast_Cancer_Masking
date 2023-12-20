from sql_api import SqlConnect
import os


#######################################
#### TO BE USED AS AN EXAMPLE ONLY ####
#######################################

sql = SqlConnect(server="ctrl-alt-elite.database.windows.net", database="ai_brestcancer", username="ctrl-alt-elite",
                 password="Tsnte7TF6nMZTPY")

sql.connect()
path = "C:/Users/chris/OneDrive - Imperial College London/CBIS Dataset/manifest-Egq0PU078220738724328010106/CBIS-DDSM/Grouped_nodir"

folder_names = os.listdir(path)
values = []
id = 0
coulmns = ("id", "patient_id", "direction", "image_view", "density", "pixel_data")

for folder in folder_names:
    id += 1
    folder_value = {}
    with open(path+"/"+folder, 'rb') as file:
        folder_value['pixel_data'] = file.read()
    folder_value['id'] = id
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
    print(id)

    sql.insert("dbo.ddsm_dataset", [folder_value], coulmns)


sql.close()
